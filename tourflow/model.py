"""领域静态结构：场馆、线路、接驳车次与分期推广安排。

只表达资料契约中的静态事实与直接推导，不做统计估计。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .materials import Materials


def parse_date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def parse_moment(text: str) -> dt.datetime:
    """解析带 Z 的 UTC 时点。"""
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Site:
    id: str
    daily_capacity: int
    protected_limit: int
    region: str


@dataclass(frozen=True)
class Route:
    id: str
    sites: tuple[str, ...]
    bus: str
    bus_seats: int


@dataclass(frozen=True)
class Campaign:
    id: str
    origin: str
    start: dt.date
    routes: tuple[str, ...]
    spillover_origins: tuple[str, ...]
    intensity: float


@dataclass(frozen=True)
class Structure:
    sites: dict[str, Site]
    routes: dict[str, Route]
    campaigns: dict[str, Campaign]
    # 每个客源地的推广（资料中一一对应）
    campaign_by_origin: dict[str, Campaign]
    # 场馆 -> 接驳车次（由线路静态推导；同一场馆只属于一趟车）
    bus_by_site: dict[str, str]
    # 车次的单日座位容量
    bus_seats: dict[str, int]

    def promoted_routes(self, origin: str) -> tuple[str, ...]:
        campaign = self.campaign_by_origin.get(origin)
        return campaign.routes if campaign else ()

    def spillover_exposure(self) -> dict[str, dt.date]:
        """每个客源地首次被任何已启动推广的外溢联系波及的日期。

        资料确认的外溢联系意味着该客源地自此不适合充当干净对照；
        没有观测到外溢不等于确认未受影响。
        """
        exposure: dict[str, dt.date] = {}
        for campaign in self.campaigns.values():
            for origin in campaign.spillover_origins:
                earlier = exposure.get(origin)
                if earlier is None or campaign.start < earlier:
                    exposure[origin] = campaign.start
        return exposure


def build_structure(materials: Materials) -> Structure:
    sites = {
        r["id"]: Site(r["id"], r["daily_capacity"], r["protected_limit"], r["region"])
        for r in materials.table("sites")
    }
    routes = {
        r["id"]: Route(r["id"], tuple(r["sites"]), r["bus"], r["bus_seats"])
        for r in materials.table("routes")
    }
    campaigns = {
        r["id"]: Campaign(
            r["id"],
            r["origin"],
            parse_date(r["start"]),
            tuple(r["routes"]),
            tuple(r["spillover_origins"]),
            r["intensity"],
        )
        for r in materials.table("campaigns")
    }
    campaign_by_origin = {c.origin: c for c in campaigns.values()}
    bus_by_site: dict[str, str] = {}
    bus_seats: dict[str, int] = {}
    for route in routes.values():
        if route.bus in bus_seats and bus_seats[route.bus] != route.bus_seats:
            raise ValueError("同一接驳车次的座位容量不一致 " + route.bus)
        bus_seats[route.bus] = route.bus_seats
        for site in route.sites:
            known = bus_by_site.get(site)
            if known is not None and known != route.bus:
                raise ValueError("场馆被多趟接驳覆盖，静态推导不唯一 " + site)
            bus_by_site[site] = route.bus
    return Structure(
        sites=sites,
        routes=routes,
        campaigns=campaigns,
        campaign_by_origin=campaign_by_origin,
        bus_by_site=bus_by_site,
        bus_seats=bus_seats,
    )
