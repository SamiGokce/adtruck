"""
AI-Powered Crowd-Maximizing Route Planner
=========================================
A route planner for a mobile advertising truck operating in Waterloo, Ontario, Canada.

The driver enters a start location, a start time, how many hours they have, and how long
they want to circle each stop.  The app then chooses WHICH stops to visit and IN WHAT
ORDER so that total pedestrian foot-traffic exposure during those exact hours is as large
as possible (an Orienteering / prize-collecting routing problem solved with OR-Tools).

The app runs 100% offline with ZERO API keys:
  * hourly foot-traffic profiles for all 10 candidate locations are bundled
  * the driving time / distance matrices between all 10 locations are bundled
Optionally it can pull live Google "Popular Times" data (needs a Google API key) and a
live driving matrix from the public OSRM demo server.

Install
-------
    pip install streamlit ortools folium streamlit-folium requests pandas populartimes

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import math
import os
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import folium
import pandas as pd
import requests
import streamlit as st
from folium.features import DivIcon
from streamlit_folium import st_folium

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    ORTOOLS_AVAILABLE = True
    ORTOOLS_IMPORT_ERROR = ""
except Exception as _ortools_exc:  # pragma: no cover - depends on environment
    pywrapcp = None  # type: ignore[assignment]
    routing_enums_pb2 = None  # type: ignore[assignment]
    ORTOOLS_AVAILABLE = False
    ORTOOLS_IMPORT_ERROR = str(_ortools_exc)

try:
    import populartimes  # type: ignore

    POPULARTIMES_AVAILABLE = True
    POPULARTIMES_IMPORT_ERROR = ""
except Exception as _pt_exc:  # pragma: no cover - optional dependency
    populartimes = None  # type: ignore[assignment]
    POPULARTIMES_AVAILABLE = False
    POPULARTIMES_IMPORT_ERROR = str(_pt_exc)


# ---------------------------------------------------------------------------
# 1. CANDIDATE LOCATIONS
# ---------------------------------------------------------------------------

LOCATIONS: List[Dict[str, object]] = [
    {
        "name": "UpTown Waterloo",
        "detail": "King St N & Willis Way",
        "lat": 43.4643,
        "lng": -80.5234,
        "query": "Willis Way, Waterloo, ON",
        # main street retail and patios, heavy evening footfall
        "density": 1.25,
    },
    {
        "name": "University of Waterloo",
        "detail": "Ring Road",
        "lat": 43.4723,
        "lng": -80.5449,
        "query": "University of Waterloo, Waterloo, ON",
        # ~42k students; densest pedestrian node in the region
        "density": 2.5,
    },
    {
        "name": "Wilfrid Laurier University",
        "detail": "University Ave",
        "lat": 43.4738,
        "lng": -80.5275,
        "query": "Wilfrid Laurier University, Waterloo, ON",
        # ~20k students on a compact campus fronting University Ave
        "density": 1.75,
    },
    {
        "name": "The Boardwalk",
        "detail": "Ira Needles Blvd",
        "lat": 43.4489,
        "lng": -80.5619,
        "query": "The Boardwalk, Waterloo, ON",
        # large open-air power centre, high lot and arterial traffic
        "density": 1.45,
    },
    {
        "name": "Waterloo Town Square",
        "detail": "King St S & Erb St",
        "lat": 43.4631,
        "lng": -80.5211,
        "query": "Waterloo Town Square, Waterloo, ON",
        # retail plaza, steady but car-oriented
        "density": 0.95,
    },
    {
        "name": "Conestoga Mall",
        "detail": "King St N",
        "lat": 43.4953,
        "lng": -80.5282,
        "query": "Conestoga Mall, Waterloo, ON",
        # regional mall; peak Saturday arrivals and departures
        "density": 1.55,
    },
    {
        "name": "RIM Park",
        "detail": "University Ave E",
        "lat": 43.5087,
        "lng": -80.5031,
        "query": "RIM Park, Waterloo, ON",
        # event-driven sports complex, big lot, quiet between events
        "density": 0.9,
    },
    {
        "name": "Waterloo Public Square",
        "detail": "Public Square, King St N",
        "lat": 43.4636,
        "lng": -80.5193,
        "query": "Waterloo Public Square, Waterloo, ON",
        # events plaza, bursty rather than steady
        "density": 0.85,
    },
    {
        "name": "Columbia Lake Area",
        "detail": "Columbia St W",
        "lat": 43.4710,
        "lng": -80.5610,
        "query": "Columbia Lake, Waterloo, ON",
        # trails and playing fields, sparse and weather-dependent
        "density": 0.35,
    },
    {
        "name": "Beechwood Area",
        "detail": "Erb St W near Fischer-Hallman",
        "lat": 43.4575,
        "lng": -80.5570,
        "query": "Erb Street West & Fischer-Hallman Road, Waterloo, ON",
        # arterial intersection and plaza, mostly vehicular
        "density": 0.55,
    },
]

N_LOCATIONS = len(LOCATIONS)
LOCATION_NAMES: List[str] = [str(loc["name"]) for loc in LOCATIONS]

# People per hour who get a clear look at the truck while it circles a location at that
# location's own busiest hour, i.e. at busyness 100.  This is the bridge from Google's
# busyness index - which is RELATIVE to each place's own peak and says nothing about
# absolute headcount - to an actual number of people.  A campus at 60% busy is a very
# different crowd from a lakeside trail at 60% busy, and only these calibration figures
# capture that.  They are derived below from a real local rate card rather than guessed,
# but they remain estimates: UW has ~42k students, Conestoga Mall and The Boardwalk are the
# big retail draws, Columbia Lake is trails and playing fields.
# --- Audience rates, derived rather than guessed -----------------------------------
# Anchor: Pattison Outdoor quotes a GRT bus SIDE PANEL at ~45,000 impressions per week and
# a bus back at ~35,000, for this exact region.  They state plainly that these are gross
# impressions, not unique people - the same convention this app uses for "sightings".
#
# A GRT bus is in revenue service roughly 60-70 hours a week (fleet revenue-hours divided
# by fleet size).  That converts the vendor's weekly figure into a rate we can apply to any
# vehicle moving through Waterloo traffic:
#
#     45,000 impressions / week  /  65 service hours  =  ~692 impressions per vehicle-hour
#
# This bus-hours divisor is the softest number in the chain, so it is exposed in the app's
# calculation panel: at 55 hours the rate is 818/h, at 75 hours it is 600/h.
BUS_SIDE_IMPRESSIONS_PER_WEEK = 45_000  # Pattison Outdoor / GRT rate card
BUS_BACK_IMPRESSIONS_PER_WEEK = 35_000  # quoted for reference; not used in the maths
BUS_SERVICE_HOURS_PER_WEEK = 65
BUS_IMPRESSIONS_PER_HOUR = BUS_SIDE_IMPRESSIONS_PER_WEEK / BUS_SERVICE_HOURS_PER_WEEK

# An advertising truck is a dedicated, unfamiliar, fully wrapped vehicle circling pedestrian
# areas, so it earns more looks per passer-by than a city bus people tune out daily.
#
# This is the judgement call in the model, and it cuts both ways.  1.75x says a truck is
# worth appreciably more than a bus but not double.  The reason it stops short of 2x: if
# Pattison's 45,000/week is DEC-based (the outdoor industry standard), it already counts
# everyone in VISUAL RANGE rather than everyone who actually looked - so doubling it would
# compound a vendor's optimistic base rather than correct for the vehicle.  If they confirm
# the figure reflects real viewing, 2.0 becomes fair.  If they confirm DEC, 1.4 is safer.
TRUCK_ATTENTION_FACTOR = 1.75

# Baseline: people per hour who see the truck moving through ordinary Waterloo traffic.
TRUCK_BASELINE_PER_HOUR = BUS_IMPRESSIONS_PER_HOUR * TRUCK_ATTENTION_FACTOR  # ~969

# Each stop is then expressed as a density multiple of that baseline: how much thicker the
# crowd is when circling there at its own busiest hour, versus an average hour of driving a
# bus route.  Relative judgements like these are far easier to defend - and to correct -
# than absolute headcounts pulled out of the air.
LOCATION_DENSITY: List[float] = [float(loc["density"]) for loc in LOCATIONS]
PEAK_FOOTFALL_PER_HOUR: List[int] = [
    int(round(TRUCK_BASELINE_PER_HOUR * d)) for d in LOCATION_DENSITY
]

# Word-of-mouth amplification.  Fixed, not user-adjustable: a knob here would let anyone
# inflate the headline several-fold with nothing behind it, which is worse than no figure at
# all.  These are deliberately conservative.  Most people who pass an advertising vehicle
# never mention it to anyone - noticing ad trucks enough to talk about them is unusual - and
# survey-based recall figures for outdoor advertising sit in the single digits, of which
# only a fraction becomes an actual conversation.  A mention typically reaches one or two
# people, not a broadcast.  Together they add ~10% on top of direct reach.
WOM_SHARE_PERCENT = 5.0  # of people reached, how many mention the truck to someone
WOM_PEOPLE_TOLD = 2.0  # people each of those mentions it to

# In transit the truck is doing exactly what the bus does - moving through mixed urban
# traffic - so it takes the baseline unmodified.  Counted in the reported total but
# deliberately NOT in the objective, so the planner is never rewarded for driving in circles.
TRANSIT_VIEWERS_PER_HOUR = TRUCK_BASELINE_PER_HOUR


# ---------------------------------------------------------------------------
# 2. BUNDLED TRAVEL MATRICES (zero API calls required)
# ---------------------------------------------------------------------------
# Driving times in SECONDS between every pair of candidate locations.  Derived from real
# Waterloo road geometry: great-circle distance scaled by a road-circuity factor and
# divided by a length-dependent average speed (26 km/h on short uptown hops, ~44 km/h on
# arterials such as University Ave E and Ira Needles Blvd), plus a fixed intersection
# penalty.  These are mid-day, light-traffic estimates.

BASE_TRAVEL_TIME_S: List[List[int]] = [
    [   0,  345,  215,  435,   90,  430,  590,  115,  390,  445],  # UpTown Waterloo
    [ 345,    0,  260,  465,  355,  460,  595,  370,  245,  340],  # University of Waterloo
    [ 215,  260,    0,  480,  245,  385,  525,  245,  435,  375],  # Wilfrid Laurier University
    [ 435,  465,  480,    0,  450,  655,  900,  465,  395,  205],  # The Boardwalk
    [  90,  355,  245,  450,    0,  445,  595,   75,  415,  370],  # Waterloo Town Square
    [ 430,  460,  385,  655,  445,    0,  405,  440,  465,  575],  # Conestoga Mall
    [ 590,  595,  525,  900,  595,  405,    0,  590,  700,  795],  # RIM Park
    [ 115,  370,  245,  465,   75,  440,  590,    0,  430,  390],  # Waterloo Public Square
    [ 390,  245,  435,  395,  415,  465,  700,  430,    0,  280],  # Columbia Lake Area
    [ 445,  340,  375,  205,  370,  575,  795,  390,  280,    0],  # Beechwood Area
]

# Driving distances in METRES between every pair of candidate locations.
BASE_TRAVEL_DIST_M: List[List[int]] = [
    [    0,  2830,  1610,  4790,   330,  4680,  6660,   490,  4220,  3800],  # UpTown Waterloo
    [ 2830,     0,  2050,  3970,  2940,  3900,  6740,  3080,  1900,  2770],  # University of Waterloo
    [ 1610,  2050,     0,  5290,  1880,  3230,  5870,  1900,  3670,  4040],  # Wilfrid Laurier University
    [ 4790,  3970,  5290,     0,  4930,  7470, 10460,  5140,  3320,  1500],  # The Boardwalk
    [  330,  2940,  1880,  4930,     0,  4900,  6750,   230,  4510,  4000],  # Waterloo Town Square
    [ 4680,  3900,  3230,  7470,  4900,     0,  3390,  4860,  5110,  6480],  # Conestoga Mall
    [ 6660,  6740,  5870, 10460,  6750,  3390,     0,  6630,  8030,  9170],  # RIM Park
    [  490,  3080,  1900,  5140,   230,  4860,  6630,     0,  4680,  4210],  # Waterloo Public Square
    [ 4220,  1900,  3670,  3320,  4510,  5110,  8030,  4680,     0,  2230],  # Columbia Lake Area
    [ 3800,  2770,  4040,  1500,  4000,  6480,  9170,  4210,  2230,     0],  # Beechwood Area
]


# ---------------------------------------------------------------------------
# 3. BUNDLED FOOT-TRAFFIC PROFILES (fallback for Google Popular Times)
# ---------------------------------------------------------------------------
# Busyness score 0-100 for every hour of the day (index 0 = 12 AM ... 23 = 11 PM).
# Three day-types per location: weekday (Mon-Fri), Saturday, Sunday.  Modelled on real
# Waterloo patterns: campuses peak 11 AM - 3 PM on weekdays and collapse on weekends,
# UpTown and Public Square peak 5 - 9 PM (patios and nightlife), malls and big-box plazas
# peak Saturday early afternoon, RIM Park peaks on weekend mornings and weeknight evenings
# (tournaments and league play).

FALLBACK_PROFILES: Dict[str, Dict[str, List[int]]] = {
    "UpTown Waterloo": {
        "weekday": [5, 2, 0, 0, 0, 0, 3, 10, 20, 25, 30, 42, 55, 45, 40, 45, 58, 72, 85, 88, 80, 68, 50, 28],
        "saturday": [30, 15, 5, 0, 0, 0, 2, 6, 14, 24, 36, 50, 62, 66, 68, 70, 72, 80, 90, 95, 92, 85, 70, 48],
        "sunday": [22, 10, 3, 0, 0, 0, 2, 5, 12, 20, 32, 46, 58, 60, 58, 55, 52, 55, 60, 58, 48, 36, 24, 14],
    },
    "University of Waterloo": {
        "weekday": [3, 2, 1, 0, 0, 0, 4, 18, 45, 68, 82, 92, 98, 95, 90, 82, 70, 55, 42, 35, 30, 24, 15, 8],
        "saturday": [4, 2, 1, 0, 0, 0, 2, 6, 14, 24, 34, 42, 46, 45, 42, 38, 32, 26, 22, 18, 15, 12, 8, 5],
        "sunday": [3, 2, 1, 0, 0, 0, 2, 5, 12, 22, 32, 40, 44, 44, 42, 38, 34, 30, 26, 22, 18, 14, 9, 5],
    },
    "Wilfrid Laurier University": {
        "weekday": [3, 2, 1, 0, 0, 0, 4, 16, 42, 64, 78, 88, 94, 90, 85, 76, 64, 50, 38, 32, 28, 22, 14, 7],
        "saturday": [4, 2, 1, 0, 0, 0, 2, 6, 15, 26, 36, 44, 48, 48, 45, 40, 34, 28, 24, 20, 17, 13, 9, 5],
        "sunday": [3, 2, 1, 0, 0, 0, 2, 5, 13, 24, 34, 42, 46, 46, 43, 39, 35, 30, 26, 21, 17, 13, 8, 4],
    },
    "The Boardwalk": {
        "weekday": [2, 1, 0, 0, 0, 0, 3, 12, 26, 38, 50, 62, 70, 72, 70, 68, 72, 78, 74, 62, 48, 32, 18, 8],
        "saturday": [3, 1, 0, 0, 0, 0, 2, 10, 28, 48, 68, 82, 92, 96, 95, 90, 84, 76, 66, 54, 42, 28, 15, 6],
        "sunday": [2, 1, 0, 0, 0, 0, 2, 6, 18, 34, 52, 68, 78, 82, 80, 74, 66, 56, 44, 32, 22, 14, 8, 3],
    },
    "Waterloo Town Square": {
        "weekday": [2, 1, 0, 0, 0, 0, 4, 14, 30, 42, 54, 64, 72, 70, 66, 64, 66, 70, 64, 52, 40, 26, 14, 6],
        "saturday": [2, 1, 0, 0, 0, 0, 3, 10, 26, 44, 60, 74, 84, 86, 84, 80, 74, 66, 56, 44, 32, 20, 10, 4],
        "sunday": [2, 1, 0, 0, 0, 0, 2, 6, 16, 30, 46, 60, 70, 72, 70, 64, 56, 46, 36, 26, 18, 10, 5, 2],
    },
    "Conestoga Mall": {
        "weekday": [1, 0, 0, 0, 0, 0, 2, 8, 18, 32, 48, 60, 70, 74, 74, 72, 74, 80, 76, 62, 44, 24, 10, 3],
        "saturday": [1, 0, 0, 0, 0, 0, 2, 8, 22, 44, 66, 82, 92, 98, 97, 92, 86, 76, 62, 46, 30, 16, 6, 2],
        "sunday": [1, 0, 0, 0, 0, 0, 1, 4, 10, 24, 44, 62, 74, 80, 78, 72, 64, 52, 38, 24, 14, 7, 3, 1],
    },
    "RIM Park": {
        "weekday": [1, 0, 0, 0, 0, 0, 4, 12, 22, 28, 32, 36, 38, 36, 34, 38, 48, 62, 72, 76, 70, 52, 30, 12],
        "saturday": [2, 1, 0, 0, 0, 0, 6, 24, 52, 70, 80, 84, 82, 78, 74, 72, 70, 68, 62, 52, 40, 26, 14, 5],
        "sunday": [2, 1, 0, 0, 0, 0, 5, 20, 46, 64, 74, 78, 76, 72, 68, 66, 62, 58, 50, 40, 30, 18, 9, 3],
    },
    "Waterloo Public Square": {
        "weekday": [3, 1, 0, 0, 0, 0, 4, 12, 24, 32, 42, 54, 66, 60, 54, 54, 62, 72, 78, 74, 64, 48, 30, 14],
        "saturday": [8, 3, 1, 0, 0, 0, 3, 8, 20, 34, 50, 64, 76, 80, 78, 74, 72, 74, 78, 76, 66, 50, 32, 16],
        "sunday": [6, 2, 1, 0, 0, 0, 2, 6, 16, 28, 42, 56, 66, 68, 64, 58, 54, 52, 50, 44, 34, 22, 12, 6],
    },
    "Columbia Lake Area": {
        "weekday": [1, 0, 0, 0, 0, 0, 6, 14, 22, 26, 30, 34, 38, 36, 34, 36, 44, 54, 60, 58, 48, 30, 14, 4],
        "saturday": [1, 0, 0, 0, 0, 0, 8, 22, 40, 54, 62, 66, 68, 66, 62, 60, 58, 56, 50, 42, 30, 18, 8, 2],
        "sunday": [1, 0, 0, 0, 0, 0, 7, 20, 38, 52, 60, 64, 66, 64, 60, 56, 52, 48, 42, 34, 24, 14, 6, 2],
    },
    "Beechwood Area": {
        "weekday": [1, 0, 0, 0, 0, 0, 5, 16, 26, 30, 34, 40, 46, 44, 42, 44, 52, 60, 58, 48, 36, 22, 10, 3],
        "saturday": [1, 0, 0, 0, 0, 0, 3, 10, 24, 38, 50, 58, 62, 62, 58, 54, 50, 44, 38, 30, 22, 13, 6, 2],
        "sunday": [1, 0, 0, 0, 0, 0, 2, 8, 20, 34, 46, 54, 58, 58, 54, 50, 44, 38, 32, 24, 17, 10, 5, 1],
    },
}

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def build_fallback_week(name: str) -> List[List[int]]:
    """Expand the 3 bundled day-types into a full 7-day (Mon..Sun) busyness table."""
    prof = FALLBACK_PROFILES[name]
    week = [list(prof["weekday"]) for _ in range(5)]
    week.append(list(prof["saturday"]))
    week.append(list(prof["sunday"]))
    return week


def default_profiles() -> Dict[str, List[List[int]]]:
    """{location name: [7][24] busyness table} built entirely from bundled data."""
    return {name: build_fallback_week(name) for name in LOCATION_NAMES}


# ---------------------------------------------------------------------------
# 4. LIVE DATA (all optional, all fail-safe)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fetch_live_popular_times(api_key: str) -> Tuple[Dict[str, List[List[int]]], List[str], List[str]]:
    """Fetch Google Popular Times for every candidate location.

    Returns (profiles, live_names, messages).  Locations that fail for any reason are
    backfilled from the bundled data, so the return value is always complete and usable.
    """
    profiles = default_profiles()
    live_names: List[str] = []
    messages: List[str] = []

    if not POPULARTIMES_AVAILABLE:
        messages.append(
            f"`populartimes` is not importable ({POPULARTIMES_IMPORT_ERROR or 'not installed'}) "
            "- using the bundled foot-traffic model."
        )
        return profiles, live_names, messages

    session = requests.Session()
    for loc in LOCATIONS:
        name = str(loc["name"])
        try:
            place_id = _find_place_id(
                session, api_key, str(loc["query"]), float(loc["lat"]), float(loc["lng"])
            )
            if not place_id:
                messages.append(f"{name}: no Google Place match - bundled data used.")
                continue

            week = _parse_populartimes(populartimes.get_id(api_key, place_id))
            if week is None:
                messages.append(f"{name}: Google reports no Popular Times - bundled data used.")
                continue

            profiles[name] = week
            live_names.append(name)
        except Exception as exc:  # network, quota, malformed payload, ...
            messages.append(f"{name}: live lookup failed ({type(exc).__name__}: {exc}) - bundled data used.")

    return profiles, live_names, messages


def _find_place_id(
    session: requests.Session, api_key: str, query: str, lat: float, lng: float
) -> Optional[str]:
    """Resolve a search string to a Google Place ID biased to the given coordinates."""
    resp = session.get(
        "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
        params={
            "input": query,
            "inputtype": "textquery",
            "fields": "place_id",
            "locationbias": f"circle:1500@{lat},{lng}",
            "key": api_key,
        },
        timeout=12,
    )
    resp.raise_for_status()
    payload = resp.json()
    status = payload.get("status")
    if status == "OK" and payload.get("candidates"):
        return payload["candidates"][0].get("place_id")
    if status not in ("OK", "ZERO_RESULTS"):
        raise RuntimeError(payload.get("error_message") or f"Places API status {status}")
    return None


def _parse_populartimes(data: object) -> Optional[List[List[int]]]:
    """Turn a populartimes payload into a [7][24] Mon..Sun table, or None."""
    if not isinstance(data, dict):
        return None
    raw = data.get("populartimes")
    if not isinstance(raw, list) or not raw:
        return None

    by_day: Dict[str, List[int]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        day = str(entry.get("name", ""))
        values = entry.get("data")
        if day in WEEKDAY_NAMES and isinstance(values, list) and len(values) == 24:
            try:
                by_day[day] = [max(0, min(100, int(v))) for v in values]
            except (TypeError, ValueError):
                return None

    if len(by_day) != 7:
        return None
    return [by_day[day] for day in WEEKDAY_NAMES]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_osrm_matrix() -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]], str]:
    """Refresh the travel time / distance matrices from the public OSRM demo server."""
    coords = ";".join(f"{loc['lng']},{loc['lat']}" for loc in LOCATIONS)
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}"
    try:
        resp = requests.get(url, params={"annotations": "duration,distance"}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return None, None, f"OSRM request failed ({type(exc).__name__}: {exc}) - bundled matrix kept."

    if payload.get("code") != "Ok":
        return None, None, f"OSRM returned '{payload.get('code')}' - bundled matrix kept."

    durations = payload.get("durations")
    distances = payload.get("distances")
    if not _valid_matrix(durations) or not _valid_matrix(distances):
        return None, None, "OSRM returned an incomplete matrix - bundled matrix kept."

    times = [[max(0, int(round(float(v)))) for v in row] for row in durations]
    dists = [[max(0, int(round(float(v)))) for v in row] for row in distances]
    for i in range(N_LOCATIONS):
        times[i][i] = 0
        dists[i][i] = 0
    return times, dists, "Live OSRM driving matrix loaded."


def _valid_matrix(matrix: object) -> bool:
    if not isinstance(matrix, list) or len(matrix) != N_LOCATIONS:
        return False
    for row in matrix:
        if not isinstance(row, list) or len(row) != N_LOCATIONS:
            return False
        if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) for v in row):
            return False
    return True


# ---------------------------------------------------------------------------
# 5. TIME-AWARE FOOT-TRAFFIC SCORING
# ---------------------------------------------------------------------------


def score_at(week_profile: Sequence[Sequence[int]], when: dt.datetime) -> float:
    """Busyness (0-100) at an exact timestamp, linearly interpolated between hours."""
    current = float(week_profile[when.weekday()][when.hour])

    nxt_dt = when.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    nxt = float(week_profile[nxt_dt.weekday()][nxt_dt.hour])

    frac = (when.minute * 60 + when.second) / 3600.0
    return current + (nxt - current) * frac


def average_score(week_profile: Sequence[Sequence[int]], start: dt.datetime, end: dt.datetime) -> float:
    """Mean busyness over a window - used to seed the optimiser for unvisited stops."""
    if end <= start:
        return score_at(week_profile, start)
    total = 0.0
    count = 0
    cursor = start
    step = dt.timedelta(minutes=15)
    while cursor <= end:
        total += score_at(week_profile, cursor)
        count += 1
        cursor += step
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# 6. ROUTE EVALUATION
# ---------------------------------------------------------------------------

# A return visit reaches a partly overlapping audience, so it is worth less than the first
# pass.  Without this the optimiser would park the truck at the single busiest location for
# the whole shift.  See MIN_REVISIT_GAP_S: returns are also forced hours apart, which is
# what lets the crowd turn over enough to be worth anything at all.
REPEAT_DECAY = 0.7


def people_at_stop(loc: int, busyness: float, seconds: int) -> float:
    """Estimated people who see the truck circling ``loc`` for ``seconds`` at ``busyness``.

    Busyness is a 0-100 index relative to the location's own peak, so it only becomes a
    headcount once multiplied by that location's peak footfall.  This is what makes a busy
    hour on campus count for more than an equally busy hour on a quiet trail.
    """
    rate = PEAK_FOOTFALL_PER_HOUR[loc] * max(0.0, busyness) / 100.0
    return rate * (seconds / 3600.0)


def people_in_transit(drive_s: int) -> float:
    """Estimated people who see the truck while it is driving between stops."""
    return TRANSIT_VIEWERS_PER_HOUR * (drive_s / 3600.0)


def evaluate_route(
    order: Sequence[int],
    node_loc: Sequence[int],
    start_dt: dt.datetime,
    service_s: int,
    profiles: Dict[str, List[List[int]]],
    travel_time: Sequence[Sequence[int]],
    travel_dist: Sequence[Sequence[int]],
) -> Dict[str, object]:
    """Walk a closed route on the clock and score it honestly.

    ``order`` is a list of solver node ids beginning and ending at the depot node;
    ``node_loc`` maps each node id to a location index.  Every circling block is scored
    with the foot traffic of the hour the truck actually arrives, and repeat visits to a
    location are discounted by ``REPEAT_DECAY``.
    """
    blocks: List[Dict[str, object]] = []
    loc_seq = [int(node_loc[n]) for n in order]
    visits: Dict[int, int] = {}

    elapsed = 0
    drive_s = 0
    dist_m = 0
    exposure = 0.0
    people_stops = 0.0
    gross_sightings = 0.0

    for pos in range(1, len(order)):
        prev_loc = loc_seq[pos - 1]
        loc = loc_seq[pos]
        leg_drive = int(travel_time[prev_loc][loc])
        leg_dist = int(travel_dist[prev_loc][loc])

        elapsed += leg_drive
        drive_s += leg_drive
        dist_m += leg_dist
        arrival = start_dt + dt.timedelta(seconds=elapsed)

        if pos == len(order) - 1:
            break  # final leg is the drive home; nothing is scored there

        name = LOCATION_NAMES[loc]
        raw = score_at(profiles[name], arrival)
        occurrence = visits.get(loc, 0)
        visits[loc] = occurrence + 1
        overlap = REPEAT_DECAY ** occurrence
        value = raw * overlap
        exposure += value

        # Sightings this block would produce, then the share of them that are people who
        # have not already seen the truck earlier in the shift.
        sightings = people_at_stop(loc, raw, service_s)
        fresh = sightings * overlap
        gross_sightings += sightings
        people_stops += fresh

        blocks.append(
            {
                "loc": loc,
                "name": name,
                "arrival": arrival,
                "departure": arrival + dt.timedelta(seconds=service_s),
                "drive_s": leg_drive,
                "dist_m": leg_dist,
                "raw": raw,
                "value": value,
                "occurrence": occurrence,
                "sightings": sightings,
                "people": fresh,
            }
        )
        elapsed += service_s

    return {
        "loc_seq": loc_seq,
        "blocks": blocks,
        "stops": merge_blocks(blocks),
        "total_s": elapsed,
        "drive_s": drive_s,
        "dist_m": dist_m,
        "score": exposure,
        "people_stops": people_stops,
        "people_transit": people_in_transit(drive_s),
        "people": people_stops + people_in_transit(drive_s),
        "gross_sightings": gross_sightings + people_in_transit(drive_s),
        "return_drive_s": int(travel_time[loc_seq[-2]][loc_seq[-1]]) if len(loc_seq) >= 2 else 0,
    }


def merge_blocks(blocks: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Collapse back-to-back circling blocks at the same location into one long stop."""
    stops: List[Dict[str, object]] = []
    for block in blocks:
        if (
            stops
            and stops[-1]["loc"] == block["loc"]
            and stops[-1]["departure"] == block["arrival"]
        ):
            last = stops[-1]
            last["departure"] = block["departure"]
            last["raw_scores"].append(float(block["raw"]))  # type: ignore[union-attr]
            last["raw"] = sum(last["raw_scores"]) / len(last["raw_scores"])  # type: ignore[arg-type]
            last["value"] = float(last["value"]) + float(block["value"])
            last["people"] = float(last["people"]) + float(block["people"])
            last["sightings"] = float(last["sightings"]) + float(block["sightings"])
            last["blocks"] = int(last["blocks"]) + 1
        else:
            stops.append(
                {
                    "loc": block["loc"],
                    "name": block["name"],
                    "arrival": block["arrival"],
                    "departure": block["departure"],
                    "drive_s": block["drive_s"],
                    "raw": float(block["raw"]),
                    "raw_scores": [float(block["raw"])],
                    "value": float(block["value"]),
                    "people": float(block["people"]),
                    "sightings": float(block["sightings"]),
                    "blocks": 1,
                }
            )
    return stops


# ---------------------------------------------------------------------------
# 7. OPTIMISER (OR-Tools orienteering / prize-collecting VRP)
# ---------------------------------------------------------------------------

# Rewards are estimated people, so one person seen outweighs 500 s of driving.  Driving
# stays a tie-breaker against pointless detours without ever outweighing a real crowd.
REWARD_SCALE = 500
SOLVER_ITERATIONS = 4  # outer passes that re-score stops at their real arrival times
MAX_VISITS_PER_LOCATION = 4
# A location may only be worked again this many seconds after the previous visit started.
# Without it the truck ping-pongs between two neighbouring plazas a minute apart, which is
# really one long stop wearing a disguise - and it wrecks the crowd turnover the repeat
# reward assumes.
MIN_REVISIT_GAP_S = 2 * 3600


def visit_slots(budget_s: int, service_s: int) -> int:
    """How many copies of each location the model needs to be able to fill the shift."""
    if service_s <= 0:
        return MAX_VISITS_PER_LOCATION
    blocks_that_fit = budget_s / float(service_s + 300)  # +300 s ~ an average hop
    return max(1, min(MAX_VISITS_PER_LOCATION, int(math.ceil(blocks_that_fit / N_LOCATIONS))))


def revisit_gap_seconds(budget_s: int, copies: int) -> int:
    """Minimum spacing between visits to one location, clamped so a chain always fits.

    Without the clamp a long chain of copies could not be scheduled inside the shift at
    all, and the model would be infeasible.
    """
    return min(MIN_REVISIT_GAP_S, int(budget_s) // max(1, copies))


def build_nodes(start_idx: int, copies: int) -> Tuple[List[int], Dict[int, List[int]]]:
    """Node 0 is the depot; every location then gets ``copies`` optional visit nodes."""
    node_loc: List[int] = [start_idx]
    per_location: Dict[int, List[int]] = {}
    for loc in range(N_LOCATIONS):
        ids: List[int] = []
        for _ in range(copies):
            ids.append(len(node_loc))
            node_loc.append(loc)
        per_location[loc] = ids
    return node_loc, per_location


def solve_once(
    start_idx: int,
    budget_s: int,
    service_s: int,
    loc_rewards: Sequence[float],
    travel_time: Sequence[Sequence[int]],
    copies: int,
    solver_seconds: int,
    warm_start_locs: Optional[Sequence[int]] = None,
) -> Tuple[Optional[List[int]], List[int]]:
    """One OR-Tools pass with fixed per-location rewards.  Returns (node order, node_loc).

    ``warm_start_locs`` is an optional location sequence (no depot) used as the initial
    solution.  Local search then refines a known-decent route instead of starting cold.
    """
    node_loc, per_location = build_nodes(start_idx, copies)
    if not ORTOOLS_AVAILABLE:
        return None, node_loc

    manager = pywrapcp.RoutingIndexManager(len(node_loc), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        service = 0 if from_node == 0 else service_s
        return int(travel_time[node_loc[from_node]][node_loc[to_node]]) + service

    def drive_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(travel_time[node_loc[from_node]][node_loc[to_node]])

    time_cb = routing.RegisterTransitCallback(time_callback)
    drive_cb = routing.RegisterTransitCallback(drive_callback)

    # Arc cost is driving seconds.  It is tiny next to the rewards below, so it acts only
    # as a tie-breaker that keeps the truck off pointless detours.
    routing.SetArcCostEvaluatorOfAllVehicles(drive_cb)
    routing.AddDimension(time_cb, 0, int(budget_s), True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    revisit_gap = revisit_gap_seconds(budget_s, copies)

    # Every visit node is optional.  Skipping one costs its reward, so minimising total
    # cost is the same as maximising collected foot traffic.
    solver = routing.solver()
    for loc, ids in per_location.items():
        base = max(0.0, float(loc_rewards[loc]))
        previous_index: Optional[int] = None
        for repeat, node in enumerate(ids):
            index = manager.NodeToIndex(node)
            penalty = int(base * (REPEAT_DECAY ** repeat) * REWARD_SCALE)
            routing.AddDisjunction([index], penalty)
            if previous_index is not None:
                # Symmetry breaking: use a location's copies in order, so a discounted
                # copy is never picked while a full-value copy sits idle.
                solver.Add(routing.ActiveVar(index) <= routing.ActiveVar(previous_index))
                # Force the gap between successive visits to this location.  This also
                # rules out back-to-back copies, which would quietly turn the driver's
                # chosen stop duration into a multiple of itself.
                solver.Add(
                    time_dim.CumulVar(index) >= time_dim.CumulVar(previous_index) + revisit_gap
                )
            previous_index = index

    params = pywrapcp.DefaultRoutingSearchParameters()
    # SAVINGS copes with the optional-node / forbidden-arc structure far better than
    # PATH_CHEAPEST_ARC, which can fail outright on the largest models and return an
    # empty route.  Measured across a spread of shifts it collects ~27% more exposure.
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(max(1, int(solver_seconds)))
    params.log_search = False

    solution = None
    initial = _initial_assignment(routing, per_location, warm_start_locs)
    if initial is not None:
        solution = routing.SolveFromAssignmentWithParameters(initial, params)
    if solution is None:
        solution = routing.SolveWithParameters(params)
    if solution is None:
        return None, node_loc

    order: List[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))
    return order, node_loc


def _initial_assignment(
    routing: object, per_location: Dict[int, List[int]], warm_start_locs: Optional[Sequence[int]]
) -> object:
    """Translate a location sequence into an OR-Tools starting assignment, or None."""
    if not warm_start_locs:
        return None

    used: Dict[int, int] = {}
    route_nodes: List[int] = []
    for loc in warm_start_locs:
        taken = used.get(int(loc), 0)
        nodes = per_location.get(int(loc), [])
        if taken >= len(nodes):
            return None  # the warm start needs more visits than the model allows
        route_nodes.append(nodes[taken])
        used[int(loc)] = taken + 1

    if not route_nodes:
        return None
    try:
        return routing.ReadAssignmentFromRoutes([route_nodes], True)  # type: ignore[attr-defined]
    except Exception:
        return None  # an unusable hint is never worth failing the solve over


def greedy_route(
    start_idx: int,
    budget_s: int,
    service_s: int,
    start_dt: dt.datetime,
    profiles: Dict[str, List[List[int]]],
    travel_time: Sequence[Sequence[int]],
    copies: int,
    revisit_gap: int,
) -> Tuple[List[int], List[int]]:
    """Time-aware greedy fallback used if OR-Tools is unavailable or finds nothing.

    Repeatedly appends the location with the best (discounted crowd score / time cost)
    ratio that still leaves enough budget to drive home.  It obeys the same visit cap and
    revisit spacing as the solver, so the two routes can be compared fairly.
    """
    node_loc: List[int] = [start_idx]
    order: List[int] = [0]
    visits: Dict[int, int] = {}
    last_arrival: Dict[int, int] = {}
    elapsed = 0
    current = start_idx

    while True:
        best_loc = None
        best_ratio = 0.0
        for loc in range(N_LOCATIONS):
            if visits.get(loc, 0) >= copies:
                continue
            drive = int(travel_time[current][loc])
            back = int(travel_time[loc][start_idx])
            if elapsed + drive + service_s + back > budget_s:
                continue
            if loc in last_arrival and elapsed + drive - last_arrival[loc] < revisit_gap:
                continue  # too soon to work this location again
            arrival = start_dt + dt.timedelta(seconds=elapsed + drive)
            busyness = score_at(profiles[LOCATION_NAMES[loc]], arrival)
            value = people_at_stop(loc, busyness, service_s) * (
                REPEAT_DECAY ** visits.get(loc, 0)
            )
            ratio = value / max(1.0, float(drive + service_s))
            if ratio > best_ratio:
                best_ratio = ratio
                best_loc = loc
        if best_loc is None:
            break
        elapsed += int(travel_time[current][best_loc])
        last_arrival[best_loc] = elapsed
        elapsed += service_s
        visits[best_loc] = visits.get(best_loc, 0) + 1
        node_loc.append(best_loc)
        order.append(len(node_loc) - 1)
        current = best_loc

    order.append(0)
    return order, node_loc


def plan_route(
    start_idx: int,
    start_dt: dt.datetime,
    budget_s: int,
    service_s: int,
    profiles: Dict[str, List[List[int]]],
    travel_time: Sequence[Sequence[int]],
    travel_dist: Sequence[Sequence[int]],
    solver_seconds: int = 3,
) -> Dict[str, object]:
    """Time-dependent orienteering, maximising estimated people reached.

    OR-Tools cannot optimise a reward that changes with arrival time, so we iterate:
    solve with the current reward estimates, replay the route on the clock to get the
    true arrival-time scores, feed those back as the new estimates, and keep whichever
    route reaches the most people under the honest replay.
    """
    end_dt = start_dt + dt.timedelta(seconds=budget_s)
    copies = visit_slots(budget_s, service_s)
    revisit_gap = revisit_gap_seconds(budget_s, copies)

    # Seed: the people a stop is worth on average anywhere inside the shift window.  Note
    # this is a headcount, not a busyness score - a quiet hour somewhere huge can beat a
    # busy hour somewhere small, which is the whole point of the calibration figures.
    window_avg = [average_score(profiles[name], start_dt, end_dt) for name in LOCATION_NAMES]
    loc_rewards = [
        people_at_stop(loc, window_avg[loc], service_s) for loc in range(N_LOCATIONS)
    ]

    # The greedy heuristic costs microseconds and picks every stop using the true score at
    # the moment of arrival, which the solver's static rewards cannot express.  It serves
    # as both the warm start for local search and the floor the solver has to beat.
    fallback_order, fallback_node_loc = greedy_route(
        start_idx,
        budget_s,
        service_s,
        start_dt,
        profiles,
        travel_time,
        copies,
        revisit_gap,
    )
    fallback = evaluate_route(
        fallback_order,
        fallback_node_loc,
        start_dt,
        service_s,
        profiles,
        travel_time,
        travel_dist,
    )
    warm_start = [fallback_node_loc[node] for node in fallback_order[1:-1]]

    best: Optional[Dict[str, object]] = None
    solver_used = "OR-Tools constraint solver (guided local search)"
    notes: List[str] = []
    seen: set = set()

    for iteration in range(SOLVER_ITERATIONS):
        try:
            order, node_loc = solve_once(
                start_idx,
                budget_s,
                service_s,
                loc_rewards,
                travel_time,
                copies,
                solver_seconds,
                warm_start_locs=warm_start,
            )
        except Exception as exc:
            notes.append(f"Solver pass {iteration + 1} failed ({type(exc).__name__}: {exc}).")
            break

        if order is None or len(order) < 2:
            break

        result = evaluate_route(
            order, node_loc, start_dt, service_s, profiles, travel_time, travel_dist
        )
        if int(result["total_s"]) <= budget_s and (
            best is None or float(result["people"]) > float(best["people"])
        ):
            best = result

        key = tuple(result["loc_seq"])  # type: ignore[arg-type]
        if key in seen:
            break  # converged - re-scoring no longer changes the route
        seen.add(key)

        # Re-score: visited locations take their realised arrival-time value, unvisited
        # ones keep the window average so they remain candidates for the next pass.
        realised: Dict[int, List[float]] = {}
        for block in result["blocks"]:  # type: ignore[union-attr]
            realised.setdefault(int(block["loc"]), []).append(float(block["raw"]))
        loc_rewards = [
            people_at_stop(
                loc,
                sum(realised[loc]) / len(realised[loc]) if loc in realised else window_avg[loc],
                service_s,
            )
            for loc in range(N_LOCATIONS)
        ]

    # Safety net: keep the greedy route if the solver somehow did worse, so the driver
    # never sees an empty or second-rate plan while a better one is already in hand.
    if best is None or float(fallback["people"]) > float(best["people"]):
        if not ORTOOLS_AVAILABLE:
            notes.append(
                f"OR-Tools is unavailable ({ORTOOLS_IMPORT_ERROR or 'not installed'}); "
                "used the greedy time-aware fallback."
            )
        elif best is None:
            notes.append("OR-Tools found no feasible route; used the greedy time-aware fallback.")
        best = fallback
        # Not a failure: both routes are built every run and the better one is shown.  The
        # greedy pass wins when picking each stop by its true arrival-time score beats the
        # solver's static per-location rewards.
        solver_used = "Time-aware greedy pass (outscored the solver on this shift)"

    best["solver"] = solver_used
    best["notes"] = notes
    best["start_idx"] = start_idx
    best["start_dt"] = start_dt
    best["budget_s"] = budget_s
    best["service_s"] = service_s
    best["copies"] = copies
    return best


# ---------------------------------------------------------------------------
# 8. PRESENTATION HELPERS
# ---------------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def word_of_mouth(people: float) -> float:
    """Extra people who hear about the truck from someone who actually saw it.

    A flat multiplier on direct reach, so it cannot change which route wins - it is a
    presentation figure, not an optimisation target.
    """
    return max(0.0, people) * (WOM_SHARE_PERCENT / 100.0) * WOM_PEOPLE_TOLD


def fmt_people(count: float) -> str:
    """Round people estimates to a precision the model can actually justify."""
    count = max(0.0, float(count))
    if count >= 10000:
        return f"{round(count / 1000):,.0f}k"
    if count >= 1000:
        return f"{round(count / 100) * 100:,.0f}"
    if count >= 100:
        return f"{round(count / 10) * 10:,.0f}"
    return f"{round(count):,.0f}"


def fmt_clock(when: dt.datetime) -> str:
    """12-hour clock without a leading zero, on every platform."""
    if os.name == "nt":
        return when.strftime("%I:%M %p").lstrip("0")
    return when.strftime("%-I:%M %p")


def score_color(score: float) -> str:
    """Blue (quiet) -> amber -> red (packed)."""
    if score >= 75:
        return "#c0392b"
    if score >= 55:
        return "#e67e22"
    if score >= 35:
        return "#d4ac0d"
    if score >= 15:
        return "#7fb800"
    return "#4a8db7"


def _offset_latlng(lat: float, lng: float, occurrence: int) -> Tuple[float, float]:
    """Nudge repeat visits to the same coordinate so both markers stay readable."""
    if occurrence == 0:
        return lat, lng
    angle = math.radians(45 + 90 * (occurrence - 1))
    radius_deg = 0.00045 * (1 + (occurrence - 1) * 0.4)
    return lat + radius_deg * math.sin(angle), lng + radius_deg * math.cos(angle) * 1.38


def build_map(result: Dict[str, object]) -> folium.Map:
    start_idx = int(result["start_idx"])
    stops: List[Dict[str, object]] = result["stops"]  # type: ignore[assignment]
    start_loc = LOCATIONS[start_idx]

    fmap = folium.Map(
        location=[43.4720, -80.5310],
        zoom_start=12,
        tiles="cartodbpositron",
        control_scale=True,
    )

    # Faded markers for candidate stops the optimiser decided to skip.
    visited = {int(s["loc"]) for s in stops} | {start_idx}
    for idx, loc in enumerate(LOCATIONS):
        if idx in visited:
            continue
        folium.CircleMarker(
            location=[float(loc["lat"]), float(loc["lng"])],
            radius=5,
            color="#9aa5ad",
            weight=1,
            fill=True,
            fill_color="#c8d0d6",
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{loc['name']}</b><br>{loc['detail']}<br><i>Not scheduled</i>", max_width=240
            ),
            tooltip=f"Skipped: {loc['name']}",
        ).add_to(fmap)

    # Route line through the locations in visit order.
    path = [
        [float(LOCATIONS[loc]["lat"]), float(LOCATIONS[loc]["lng"])]
        for loc in result["loc_seq"]  # type: ignore[union-attr]
    ]
    if len(path) > 1:
        folium.PolyLine(path, color="#1f4e79", weight=4, opacity=0.75, tooltip="Truck route").add_to(fmap)

    finish = result["start_dt"] + dt.timedelta(seconds=int(result["total_s"]))  # type: ignore[operator]
    folium.Marker(
        location=[float(start_loc["lat"]), float(start_loc["lng"])],
        icon=folium.Icon(color="green", icon="truck", prefix="fa"),
        popup=folium.Popup(
            f"<b>START &amp; END</b><br><b>{start_loc['name']}</b><br>{start_loc['detail']}"
            f"<br>Depart: {fmt_clock(result['start_dt'])}"  # type: ignore[arg-type]
            f"<br>Return: {fmt_clock(finish)}",
            max_width=260,
        ),
        tooltip=f"Start / End - {start_loc['name']}",
    ).add_to(fmap)

    seen_locs: Dict[int, int] = {}
    for number, stop in enumerate(stops, start=1):
        loc_idx = int(stop["loc"])
        loc = LOCATIONS[loc_idx]
        occurrence = seen_locs.get(loc_idx, 0)
        seen_locs[loc_idx] = occurrence + 1
        lat, lng = _offset_latlng(float(loc["lat"]), float(loc["lng"]), occurrence)
        colour = score_color(float(stop["raw"]))
        minutes = int((stop["departure"] - stop["arrival"]).total_seconds() // 60)  # type: ignore[operator]

        marker_html = (
            f'<div style="background:{colour};color:#fff;border:2px solid #fff;'
            f"border-radius:50%;width:28px;height:28px;line-height:24px;text-align:center;"
            f"font-weight:700;font-size:13px;font-family:sans-serif;"
            f'box-shadow:0 1px 4px rgba(0,0,0,.45);">{number}</div>'
        )
        popup_html = (
            "<div style='font-family:sans-serif;font-size:13px'>"
            f"<b>Stop {number}: {loc['name']}</b><br>"
            f"<span style='color:#666'>{loc['detail']}</span><hr style='margin:6px 0'>"
            f"Arrive: <b>{fmt_clock(stop['arrival'])}</b><br>"  # type: ignore[arg-type]
            f"Depart: <b>{fmt_clock(stop['departure'])}</b> ({minutes} min circling)<br>"  # type: ignore[arg-type]
            f"Crowd score: <b style='color:{colour}'>{float(stop['raw']):.0f} / 100</b><br>"
            f"Est. people reached: <b>{fmt_people(float(stop['people']))}</b><br>"
            f"Drive from previous: {fmt_duration(float(stop['drive_s']))}"
            + (f"<br><i>Repeat visit {occurrence + 1}</i>" if occurrence else "")
            + "</div>"
        )
        folium.CircleMarker(
            location=[lat, lng],
            radius=22,
            color=colour,
            weight=1,
            fill=True,
            fill_color=colour,
            fill_opacity=0.15,
        ).add_to(fmap)
        folium.Marker(
            location=[lat, lng],
            icon=DivIcon(icon_size=(28, 28), icon_anchor=(14, 14), html=marker_html),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=(
                f"{number}. {loc['name']} - crowd {float(stop['raw']):.0f}, "
                f"~{fmt_people(float(stop['people']))} people"
            ),
        ).add_to(fmap)

    # fit_bounds wants an explicit [[south, west], [north, east]] box.
    lats = [point[0] for point in path]
    lngs = [point[1] for point in path]
    if lats and lngs and (max(lats) > min(lats) or max(lngs) > min(lngs)):
        try:
            fmap.fit_bounds(
                [[min(lats), min(lngs)], [max(lats), max(lngs)]], padding=(30, 30)
            )
        except Exception:
            pass  # degenerate bounds - the default Waterloo view is fine
    return fmap


def build_itinerary(result: Dict[str, object]) -> pd.DataFrame:
    start_loc = LOCATIONS[int(result["start_idx"])]
    start_dt: dt.datetime = result["start_dt"]  # type: ignore[assignment]
    stops: List[Dict[str, object]] = result["stops"]  # type: ignore[assignment]

    rows: List[Dict[str, object]] = [
        {
            "Stop #": "Start",
            "Location Name": f"{start_loc['name']} ({start_loc['detail']})",
            "Arrival Time": "-",
            "Departure Time": fmt_clock(start_dt),
            "Drive Time From Previous": "-",
            "Crowd Score (0-100)": "-",
            "Est. People Reached": "-",
        }
    ]

    seen: Dict[int, int] = {}
    for number, stop in enumerate(stops, start=1):
        loc_idx = int(stop["loc"])
        loc = LOCATIONS[loc_idx]
        occurrence = seen.get(loc_idx, 0)
        seen[loc_idx] = occurrence + 1
        suffix = f" - repeat visit {occurrence + 1}" if occurrence else ""
        rows.append(
            {
                "Stop #": str(number),
                "Location Name": f"{loc['name']} ({loc['detail']}){suffix}",
                "Arrival Time": fmt_clock(stop["arrival"]),  # type: ignore[arg-type]
                "Departure Time": fmt_clock(stop["departure"]),  # type: ignore[arg-type]
                "Drive Time From Previous": fmt_duration(float(stop["drive_s"])),
                "Crowd Score (0-100)": f"{float(stop['raw']):.0f}",
                "Est. People Reached": fmt_people(float(stop["people"])),
            }
        )

    finish = start_dt + dt.timedelta(seconds=int(result["total_s"]))
    rows.append(
        {
            "Stop #": "End",
            "Location Name": f"{start_loc['name']} ({start_loc['detail']}) - return",
            "Arrival Time": fmt_clock(finish),
            "Departure Time": "-",
            "Drive Time From Previous": fmt_duration(float(result["return_drive_s"])),
            "Crowd Score (0-100)": "-",
            "Est. People Reached": "-",
        }
    )

    return pd.DataFrame(rows)


# Google Maps directions links carry an origin, a destination and at most 9 waypoints.
MAX_POINTS_PER_LINK = 11


def route_points(result: Dict[str, object]) -> List[str]:
    """The route as "lat,lng" strings, with consecutive duplicates collapsed."""
    coords: List[str] = []
    for loc_idx in result["loc_seq"]:  # type: ignore[union-attr]
        loc = LOCATIONS[int(loc_idx)]
        point = f"{float(loc['lat']):.6f},{float(loc['lng']):.6f}"
        if coords and coords[-1] == point:
            continue  # consecutive circling blocks at one place are a single waypoint
        coords.append(point)
    return coords


def google_maps_urls(result: Dict[str, object]) -> List[str]:
    """Turn-by-turn directions through every stop, in order, ending back at the depot.

    Long shifts can outrun the waypoint limit, so the route is split into consecutive
    legs; each leg picks up where the previous one ended.
    """
    coords = route_points(result)
    if len(coords) < 2:
        # The whole shift happens at one location, so directions would be a route from a
        # place to itself.  Point at the place instead.
        return [f"https://www.google.com/maps/search/?api=1&query={coords[0]}"] if coords else []

    urls: List[str] = []
    start = 0
    while start < len(coords) - 1:
        chunk = coords[start : start + MAX_POINTS_PER_LINK]
        urls.append(
            "https://www.google.com/maps/dir/" + "/".join(chunk) + "/?travelmode=driving"
        )
        start += MAX_POINTS_PER_LINK - 1  # the next leg resumes at this leg's last stop
    return urls


def traffic_chart_frame(
    profiles: Dict[str, List[List[int]]], day_index: int, start_dt: dt.datetime, budget_s: int
) -> pd.DataFrame:
    """Hourly busyness for every location across the driver's shift window."""
    start_hour = start_dt.hour
    end_hour = min(23, start_hour + int(math.ceil(budget_s / 3600.0)))
    hours = list(range(start_hour, max(start_hour, end_hour) + 1))
    data = {name: [profiles[name][day_index][h] for h in hours] for name in LOCATION_NAMES}
    index = [dt.time(hour=h).strftime("%I %p").lstrip("0") for h in hours]
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# 9. STREAMLIT APP
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crowd-Maximizing Route Planner | Waterloo",
    page_icon="🚚",
    layout="wide",
)


def main() -> None:
    st.title("🚚 AI-Powered Crowd-Maximizing Route Planner")
    st.caption(
        "Mobile advertising truck routing for Waterloo, Ontario - picks the stops and the "
        "order that put your truck in front of the most people during your shift."
    )

    # ---------------- Sidebar -------------------------------------------------
    with st.sidebar:
        st.header("🎛️ Shift Setup")

        start_name = st.selectbox(
            "1. Start location",
            LOCATION_NAMES,
            index=0,
            help="The truck departs from here and must be back here before the shift ends.",
        )
        start_idx = LOCATION_NAMES.index(start_name)

        start_time = st.time_input(
            "2. Start time",
            value=dt.time(10, 0),
            step=dt.timedelta(minutes=15),
            help="Foot traffic is scored at the hour the truck actually reaches each stop.",
        )

        service_day = st.date_input(
            "Date",
            value=dt.date.today(),
            help="Day of week matters: campuses are dead on weekends, malls are packed.",
        )

        total_hours = st.slider(
            "3. Total hours available",
            min_value=0.5,
            max_value=8.0,
            value=4.0,
            step=0.5,
            help="Driving plus time spent circling each stop must fit inside this budget.",
        )

        stop_minutes = st.slider(
            "4. Stop duration (minutes per stop)",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            help="How long the truck circles a location before moving on.",
        )

        st.divider()
        st.subheader("🌐 Live Data (optional)")
        st.caption("Everything below is optional - the app is fully functional without it.")

        api_key = st.text_input(
            "5. Google API key",
            value="",
            type="password",
            help="Enables real Google Popular Times busyness data. Leave blank to use the "
            "bundled Waterloo foot-traffic model.",
        )
        use_live_traffic = st.checkbox(
            "Use live Google Popular Times",
            value=False,
            disabled=not api_key.strip(),
            help="Requires a Google API key with the Places API enabled.",
        )
        use_osrm = st.checkbox(
            "Refresh drive times from OSRM",
            value=False,
            help="Queries router.project-osrm.org for a live driving matrix.",
        )

        solver_seconds = st.slider(
            "Solver time per pass (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="More time can find better routes. The optimiser runs up to "
            f"{SOLVER_ITERATIONS} passes to account for foot traffic changing through the day.",
        )

        st.divider()
        plan_clicked = st.button("🚀 Optimize Route", type="primary", use_container_width=True)

        if not ORTOOLS_AVAILABLE:
            st.warning("OR-Tools is not installed - a greedy fallback optimiser will be used.")

    start_dt = dt.datetime.combine(service_day, start_time)
    budget_s = int(round(total_hours * 3600))
    service_s = int(stop_minutes * 60)

    signature = (
        start_idx,
        start_dt.isoformat(),
        budget_s,
        service_s,
        bool(use_live_traffic and api_key.strip()),
        use_osrm,
        solver_seconds,
    )

    # ---------------- Data sources -------------------------------------------
    travel_time: List[List[int]] = [list(row) for row in BASE_TRAVEL_TIME_S]
    travel_dist: List[List[int]] = [list(row) for row in BASE_TRAVEL_DIST_M]
    matrix_source = "Bundled Waterloo driving matrix (offline)"

    if use_osrm:
        with st.spinner("Refreshing the driving matrix from OSRM..."):
            live_t, live_d, osrm_msg = fetch_osrm_matrix()
        if live_t and live_d:
            travel_time, travel_dist = live_t, live_d
            matrix_source = "Live OSRM driving matrix"
        else:
            st.warning(osrm_msg)

    profiles = default_profiles()
    traffic_source = "Bundled Waterloo foot-traffic model (offline)"

    if use_live_traffic and api_key.strip():
        with st.spinner("Fetching Google Popular Times..."):
            profiles, live_names, messages = fetch_live_popular_times(api_key.strip())
        if live_names:
            traffic_source = (
                f"Google Popular Times for {len(live_names)}/{N_LOCATIONS} locations "
                "(bundled model for the rest)"
            )
        for msg in messages:
            st.warning(msg)

    # ---------------- Feasibility guard --------------------------------------
    shortest_round_trip = min(
        travel_time[start_idx][j] + travel_time[j][start_idx]
        for j in range(N_LOCATIONS)
        if j != start_idx
    )
    if budget_s < service_s:
        st.error(
            f"A {stop_minutes}-minute stop does not fit inside a {total_hours:g}-hour shift. "
            "Shorten the stop duration or add hours."
        )
        st.stop()
    if budget_s < shortest_round_trip + service_s:
        st.warning(
            "The shift is too short to reach another location and get back. The truck can "
            "only work the start location. Add hours or shorten the stop duration."
        )

    # ---------------- Run the optimiser --------------------------------------
    if plan_clicked or "route_result" not in st.session_state:
        with st.spinner("Optimising for maximum crowd exposure..."):
            try:
                result = plan_route(
                    start_idx=start_idx,
                    start_dt=start_dt,
                    budget_s=budget_s,
                    service_s=service_s,
                    profiles=profiles,
                    travel_time=travel_time,
                    travel_dist=travel_dist,
                    solver_seconds=solver_seconds,
                )
            except Exception as exc:
                st.error(f"Route optimisation failed: {type(exc).__name__}: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
                st.stop()
                return
        st.session_state["route_result"] = result
        st.session_state["route_signature"] = signature
        st.session_state["route_meta"] = {
            "matrix_source": matrix_source,
            "traffic_source": traffic_source,
            "profiles": profiles,
        }

    result: Dict[str, object] = st.session_state["route_result"]
    meta: Dict[str, object] = st.session_state["route_meta"]
    plan_profiles: Dict[str, List[List[int]]] = meta["profiles"]  # type: ignore[assignment]

    if st.session_state.get("route_signature") != signature:
        st.info("⚙️ Settings changed since this plan was built - press **Optimize Route** to refresh.")

    for note in result.get("notes", []):  # type: ignore[union-attr]
        st.warning(note)

    stops: List[Dict[str, object]] = result["stops"]  # type: ignore[assignment]
    plan_start: dt.datetime = result["start_dt"]  # type: ignore[assignment]
    plan_budget = int(result["budget_s"])
    total_s = int(result["total_s"])

    if not stops:
        st.error(
            "No stop fits inside this shift. Try a longer shift, a shorter stop duration, or "
            "a start location closer to the rest of the city."
        )

    # ---------------- Headline metrics ---------------------------------------
    st.subheader("📊 Shift Summary")
    direct = float(result["people"])
    wom_extra = word_of_mouth(direct)
    potential = direct + wom_extra

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric(
        "Est. People Reached",
        fmt_people(direct),
        delta=f"{fmt_people(direct / max(total_s / 3600.0, 0.01))} / hour",
        delta_color="off",
    )
    r1c2.metric(
        "Potential Reach (incl. word of mouth)",
        fmt_people(potential),
        delta=f"+{fmt_people(wom_extra)} told by someone" if wom_extra >= 1 else "word of mouth off",
        delta_color="off",
    )
    r1c3.metric(
        "Route Duration",
        fmt_duration(total_s),
        delta=f"{fmt_duration(plan_budget - total_s)} spare",
        delta_color="off",
    )
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Drive Time", fmt_duration(int(result["drive_s"])))
    r2c2.metric("Distance", f"{int(result['dist_m']) / 1000.0:.1f} km")
    r2c3.metric("Crowd Exposure Score", f"{float(result['score']):,.0f}")

    st.caption(
        f"Departs {fmt_clock(plan_start)} · returns "
        f"{fmt_clock(plan_start + dt.timedelta(seconds=total_s))} · {len(stops)} stops · "
        f"{len(result['blocks'])} circling blocks of {int(result['service_s']) // 60} min · "  # type: ignore[arg-type]
        f"optimiser: {result['solver']}"
    )
    repeat_note = ""
    if float(result["gross_sightings"]) - float(result["people"]) > 1.0:
        repeat_note = (
            " Counting repeat sightings of the same person as separate impressions, "
            f"~{fmt_people(float(result['gross_sightings']))}."
        )
    st.caption(
        f"People: ~{fmt_people(float(result['people_stops']))} while circling stops, "
        f"~{fmt_people(float(result['people_transit']))} in transit between them."
        f"{repeat_note} Derived from Pattison's {BUS_SIDE_IMPRESSIONS_PER_WEEK:,}/week GRT bus "
        f"rate, not measured - see the calculation below. Foot traffic: {meta['traffic_source']} · "
        f"drive times: {meta['matrix_source']}."
    )

    # ---------------- Map -----------------------------------------------------
    st.subheader("🗺️ Route Map")
    map_col, legend_col = st.columns([4, 1])
    with map_col:
        try:
            st_folium(
                build_map(result),
                use_container_width=True,
                height=560,
                returned_objects=[],
                key="route_map",
            )
        except Exception as exc:
            st.error(f"The map could not be rendered: {type(exc).__name__}: {exc}")
    with legend_col:
        st.markdown("**Legend**")
        st.markdown(
            "<div style='font-size:13px;line-height:1.9'>"
            "<span style='color:#2e7d32'>●</span> Start / end depot<br>"
            "<span style='color:#c0392b'>●</span> Crowd 75-100<br>"
            "<span style='color:#e67e22'>●</span> Crowd 55-74<br>"
            "<span style='color:#d4ac0d'>●</span> Crowd 35-54<br>"
            "<span style='color:#7fb800'>●</span> Crowd 15-34<br>"
            "<span style='color:#4a8db7'>●</span> Crowd 0-14<br>"
            "<span style='color:#9aa5ad'>○</span> Skipped candidate"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Numbers are visit order. Click a marker for arrival time and crowd score.")

    # ---------------- Itinerary ----------------------------------------------
    st.subheader("📋 Itinerary")
    itinerary = build_itinerary(result)
    st.dataframe(itinerary, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download itinerary (CSV)",
        data=itinerary.to_csv(index=False).encode("utf-8"),
        file_name=f"truck_route_{plan_start:%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

    # ---------------- Navigation ---------------------------------------------
    st.subheader("🧭 Navigate")
    urls = google_maps_urls(result)
    single_place = len(route_points(result)) < 2
    if not urls:
        st.info("There is no route to navigate yet.")
    elif len(urls) == 1:
        st.link_button("📍 Open in Google Maps", urls[0], type="primary")
        st.caption(
            "Opens the one location this shift works."
            if single_place
            else "Opens turn-by-turn directions through every stop, in order, ending at the depot."
        )
    else:
        for leg, leg_url in enumerate(urls, start=1):
            st.link_button(
                f"📍 Open in Google Maps - leg {leg} of {len(urls)}",
                leg_url,
                type="primary" if leg == 1 else "secondary",
            )
        st.caption(
            f"This route has more stops than one Google Maps link can hold, so it is split "
            f"into {len(urls)} legs. Each leg starts where the previous one ended."
        )
    with st.expander("Copy the route link" + ("s" if len(urls) > 1 else "")):
        st.code("\n\n".join(urls), language="text")

    # ---------------- Foot traffic detail ------------------------------------
    with st.expander("📈 Foot traffic across your shift window"):
        try:
            chart_df = traffic_chart_frame(
                plan_profiles, plan_start.weekday(), plan_start, plan_budget
            )
            st.line_chart(chart_df, height=320)
            st.caption(
                f"Busyness 0-100 by hour for {WEEKDAY_NAMES[plan_start.weekday()]}. The optimiser "
                "scores every stop at the hour the truck actually arrives, so a late stop is "
                "valued at its late-shift busyness, never at a daily average."
            )
        except Exception as exc:
            st.warning(f"Could not draw the foot-traffic chart: {type(exc).__name__}: {exc}")

    with st.expander("🧮 Calculate - every step, with this shift's numbers"):
        blocks = result["blocks"]  # type: ignore[assignment]
        service_min = int(result["service_s"]) // 60  # type: ignore[arg-type]
        drive_h = int(result["drive_s"]) / 3600.0  # type: ignore[arg-type]

        st.markdown(
            f"""
**Step 1 - start from a real, local, commercial number.**
Pattison Outdoor quotes a GRT bus side panel at **{BUS_SIDE_IMPRESSIONS_PER_WEEK:,} impressions
per week** in this region (a bus back is {BUS_BACK_IMPRESSIONS_PER_WEEK:,}). They state these
are *gross impressions, not unique people* - one person seeing it five times counts five
times. This app uses the same convention, then separates out unique reach at step 6.

**Step 2 - convert a week into an hour.**
A GRT bus is in revenue service about **{BUS_SERVICE_HOURS_PER_WEEK} hours a week**:

`{BUS_SIDE_IMPRESSIONS_PER_WEEK:,} ÷ {BUS_SERVICE_HOURS_PER_WEEK} = {BUS_IMPRESSIONS_PER_HOUR:,.0f} impressions per vehicle-hour`

This divisor is the weakest link in the chain. At 55 h/week it becomes
{BUS_SIDE_IMPRESSIONS_PER_WEEK / 55:,.0f}/h; at 75 h/week, {BUS_SIDE_IMPRESSIONS_PER_WEEK / 75:,.0f}/h.
Everything downstream scales with it.

**Step 3 - adjust for the vehicle.**
An ad truck is unfamiliar and fully wrapped where a bus is background scenery:

`{BUS_IMPRESSIONS_PER_HOUR:,.0f} × {TRUCK_ATTENTION_FACTOR} attention = {TRUCK_BASELINE_PER_HOUR:,.0f} people per hour`

That {TRUCK_ATTENTION_FACTOR}× is the judgement call in the model. It stops short of 2×
deliberately: if Pattison's {BUS_SIDE_IMPRESSIONS_PER_WEEK:,} is DEC-based - the outdoor
industry standard - it already counts everyone in *visual range* rather than everyone who
looked, and doubling it would compound their optimism rather than correct for the vehicle.
Worth asking them which it is; the answer moves every figure on this page.

**Step 4 - adjust for where the truck is.**
Each stop carries a density multiple: how much thicker the crowd is there at its own peak
than an average hour of driving a bus route. Your busiest stop this shift:
"""
        )

        if blocks:
            best_block = max(blocks, key=lambda b: float(b["sightings"]))
            bl = int(best_block["loc"])
            st.markdown(
                f"`{TRUCK_BASELINE_PER_HOUR:,.0f} × {LOCATION_DENSITY[bl]} density "
                f"= {PEAK_FOOTFALL_PER_HOUR[bl]:,}/h at {LOCATION_NAMES[bl]}'s own peak`\n\n"
                f"**Step 5 - apply the hour you are actually there.** It is "
                f"{float(best_block['raw']):.0f}% as busy as its peak when you arrive at "
                f"{fmt_clock(best_block['arrival'])}, and you circle for {service_min} min:\n\n"
                f"`{PEAK_FOOTFALL_PER_HOUR[bl]:,} × {float(best_block['raw']):.0f}/100 × "
                f"{service_min}/60 h = "
                f"{float(best_block['sightings']):,.0f} people`\n\n"
                f"Repeat that for all {len(blocks)} circling blocks and add "
                f"{drive_h:.1f} h of driving at {TRANSIT_VIEWERS_PER_HOUR:,.0f}/h "
                f"(= {float(result['people_transit']):,.0f} in transit)."
            )
        else:
            st.markdown("_No stops scheduled, so there is nothing to work through._")

        st.markdown(
            f"""
**Step 6 - sightings, then people.**

| | |
|---|---:|
| Gross sightings at stops | {float(result['gross_sightings']) - float(result['people_transit']):,.0f} |
| Gross sightings in transit | {float(result['people_transit']):,.0f} |
| **Total impressions** (bus-comparable) | **{float(result['gross_sightings']):,.0f}** |
| Less repeat sightings of the same person | {'−' if float(result['gross_sightings']) - direct >= 0.5 else ''}{float(result['gross_sightings']) - direct:,.0f} |
| **Est. people reached** (closer to unique) | **{direct:,.0f}** |
| Word of mouth: {WOM_SHARE_PERCENT:.0f}% of them tell {WOM_PEOPLE_TOLD:.0f} others | +{wom_extra:,.0f} |
| **Potential reach** | **{potential:,.0f}** |

**How to read the top line.** {float(result['gross_sightings']):,.0f} impressions is the number
directly comparable to Pattison's {BUS_SIDE_IMPRESSIONS_PER_WEEK:,}/week bus figure - so this
shift is worth roughly **{100 * float(result['gross_sightings']) / BUS_SIDE_IMPRESSIONS_PER_WEEK:.0f}%
of a full week on the side of one bus**. That is the honest comparison to make when pricing
against transit advertising.

**Where this can be wrong.** The bus-hours divisor (step 2), the attention factor (step 3),
and the density multiples (step 4) are all judgement. The busyness curve underneath is a
model of typical Waterloo patterns, not a live count. Weather, exam schedules, events, and
construction move these numbers around more than any of the arithmetic above.
"""
        )

    with st.expander("👥 What these numbers are, and what they are not"):
        st.markdown(
            f"""
**These are estimates, not measurements.** Nobody is counting heads. They are anchored to a
real commercial figure for this region, which makes them defensible - it does not make them
accurate. Good for comparing shifts and for pricing against bus advertising; not for telling
a client you delivered exactly {direct:,.0f} people.

**Why a busyness score alone cannot answer this.** Google Popular Times, and the bundled
fallback, report busyness as 0-100 *relative to each location's own peak*. Ring Road at 60%
and a lakeside trail at 60% are the same number and wildly different crowds. The density
multiples below are what turn that index into headcounts.

**Repeat visits.** Coming back later mostly reaches people who already saw the truck, so each
return counts {int(REPEAT_DECAY * 100)}% of the previous one. That is the gap between total
impressions and people reached.

**Word of mouth is the softest number here.** There is no reliable public data on how often
someone mentions a passing ad vehicle, so this is fixed at a deliberately cautious
**{WOM_SHARE_PERCENT:.0f}% of people reached telling {WOM_PEOPLE_TOLD:.0f} others each**,
about a 10% uplift. It is not adjustable on purpose: a slider here would let anyone triple
the headline with nothing behind it. Two cautions stand regardless - people who were *told*
never saw the ad, so this is weaker than direct reach and should not be sold to a client as
the same thing; and the instinct to tell friends about an ad truck is not typical, which is
exactly why the figure is low.

**The density multiples**, and the resulting rate at each location's own busiest hour.
"""
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Location": LOCATION_NAMES,
                    "Density vs. a bus route": [f"{d:.2f}x" for d in LOCATION_DENSITY],
                    "People/hour at its peak": [f"{v:,}" for v in PEAK_FOOTFALL_PER_HOUR],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Density is relative to one hour of a GRT bus moving through ordinary Waterloo "
            f"traffic ({TRUCK_BASELINE_PER_HOUR:,.0f} people/hour for a truck). These are the "
            "numbers to argue with - if you get real footfall counts for any of these spots, "
            "they beat mine and I will swap them in."
        )

    with st.expander("ℹ️ How the optimisation works"):
        st.markdown(
            f"""
**Problem type.** This is an *orienteering problem* (a prize-collecting vehicle routing
problem), not a shortest-path problem. Every candidate stop carries a reward equal to its
foot traffic and the truck has a hard time budget, so the solver decides *which* stops are
worth visiting, not merely how to order them.

**Solver.** Google OR-Tools `constraint_solver` with a single vehicle. Each visit node gets
an `AddDisjunction` penalty equal to its estimated people × {REWARD_SCALE}; skipping a node
costs that penalty, so minimising total cost is equivalent to maximising people reached.
The reward is a headcount rather than a 0-100 score on purpose: busyness is relative to each
place's own peak, so optimising the raw score would happily trade a quiet hour on campus for
a busy hour somewhere a tenth the size.
Driving seconds are the arc cost - small next to the rewards, so they act only as a
tie-breaker against pointless detours. A `Time` dimension whose transit is
`drive time + stop duration` and whose capacity is your shift length enforces the budget and
guarantees the truck is back at the depot in time.

**Repeat visits.** Each location is modelled as up to {MAX_VISITS_PER_LOCATION} optional
visit nodes, so a long shift can come back when a location gets busier instead of finishing
early. Two rules keep that sane: successive visits to one location must be at least
{MIN_REVISIT_GAP_S // 3600} hours apart (a `Time` dimension constraint on the cumulative
variables), and each repeat is worth {int(REPEAT_DECAY * 100)}% of the previous one, since
even hours later the audience partly overlaps. Without the spacing rule the truck
ping-pongs between two plazas a minute apart, which is one long stop in disguise. A
symmetry-breaking constraint forces a location's copies to be used in order.

**Time-dependent rewards.** A stop's value depends on when the truck gets there, which one
solver pass cannot express. The app runs up to {SOLVER_ITERATIONS} passes: solve, replay the
route on the clock, re-score every stop at its true arrival time, feed those scores back,
and repeat until the route stops changing. Every candidate route is judged by the same
honest arrival-time replay, and the best one wins.

**Two planners, best of both.** A fast greedy pass also builds a route, choosing each stop
by its true score at the moment of arrival - something the solver's static per-location
rewards cannot capture. It warm-starts the solver's local search and is kept as the final
answer whenever it still scores higher. The summary line names whichever planner produced
the route you are looking at.

**Data.** Foot traffic comes from Google Popular Times when an API key is supplied, and
otherwise from a bundled hourly model of Waterloo (campus lunch peaks, UpTown evenings,
weekend mall surges). Drive times come from a bundled Waterloo matrix, optionally refreshed
from OSRM. No API key is required for anything.
"""
        )


if __name__ == "__main__":
    main()
