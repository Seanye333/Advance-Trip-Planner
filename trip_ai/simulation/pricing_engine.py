"""
Dynamic Pricing Prediction Engine
===================================
Forecasts price movements for:
  - Flight tickets (±40% from base over 90 days)
  - Hotel rooms  (±30% from base, weekends more expensive)
  - Attraction tickets (stable, with event-driven spikes)

Methodology:
  - Rule-based baseline (booking lead time, seasonality, day-of-week)
  - Prophet-style additive decomposition:
      price = base × trend × seasonal × lead_time × noise

In production: train Prophet or an LSTM on historical booking data from
Skyscanner / Google Flights / Expedia APIs.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from enum import Enum

import numpy as np


class PriceCategory(str, Enum):
    FLIGHT     = "flight"
    HOTEL      = "hotel"
    ATTRACTION = "attraction"


# ─── Helper curves ────────────────────────────────────────────────────────────

def _lead_time_factor(days_until_trip: int, category: PriceCategory) -> float:
    """
    How does booking lead time affect price?
    Flights: U-shaped (too early or too late → expensive)
    Hotels : monotonically increasing (later = pricier)
    """
    d = max(days_until_trip, 0)
    if category == PriceCategory.FLIGHT:
        # Optimal: 3–8 weeks ahead
        optimal = 42  # days
        deviation = abs(d - optimal) / optimal
        return 1.0 + 0.35 * deviation ** 0.7
    elif category == PriceCategory.HOTEL:
        # Last-minute surge above 80%, drop below 7 days sometimes
        if d < 3:
            return 1.25
        elif d < 7:
            return 1.10
        elif d < 14:
            return 1.0
        else:
            return max(0.85, 1.0 - 0.002 * (d - 14))
    else:
        return 1.0  # attractions: stable


def _seasonal_factor(month: int, category: PriceCategory) -> float:
    flight_seasonal = {
        1: 0.85, 2: 0.82, 3: 0.90, 4: 0.95, 5: 1.00,
        6: 1.15, 7: 1.30, 8: 1.35, 9: 1.05, 10: 0.90,
        11: 0.80, 12: 1.20,
    }
    hotel_seasonal = {
        1: 0.80, 2: 0.78, 3: 0.88, 4: 0.95, 5: 1.00,
        6: 1.10, 7: 1.25, 8: 1.30, 9: 1.05, 10: 0.92,
        11: 0.82, 12: 1.15,
    }
    if category == PriceCategory.FLIGHT:
        return flight_seasonal.get(month, 1.0)
    elif category == PriceCategory.HOTEL:
        return hotel_seasonal.get(month, 1.0)
    return 1.0


def _dow_factor(weekday: int, category: PriceCategory) -> float:
    """0=Mon, 6=Sun"""
    if category == PriceCategory.FLIGHT:
        # Cheapest: Tue, Wed; priciest: Fri, Sun
        dow = [0.95, 0.90, 0.90, 0.95, 1.05, 1.10, 1.05]
    elif category == PriceCategory.HOTEL:
        # Weekend premium
        dow = [0.95, 0.90, 0.90, 0.95, 1.05, 1.15, 1.10]
    else:
        dow = [1.0] * 7
    return dow[weekday]


class DynamicPricingEngine:
    """
    Predict prices and identify the best booking window.
    """

    def __init__(self, noise_std: float = 0.03) -> None:
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed=42)

    def predict_price(
        self,
        base_price_usd: float,
        trip_date: date,
        category: PriceCategory,
        booking_date: date | None = None,
    ) -> float:
        """
        Predict price on *trip_date* if booked on *booking_date*.
        Defaults to booking today.
        """
        booking_date = booking_date or date.today()
        days_ahead = (trip_date - booking_date).days

        lead  = _lead_time_factor(days_ahead, category)
        season = _seasonal_factor(trip_date.month, category)
        dow    = _dow_factor(trip_date.weekday(), category)
        noise  = float(self._rng.normal(1.0, self.noise_std))

        price = base_price_usd * lead * season * dow * noise
        return max(0.0, round(price, 2))

    def price_calendar(
        self,
        base_price_usd: float,
        category: PriceCategory,
        start_date: date,
        days: int = 30,
    ) -> list[dict]:
        """
        Generate a price forecast calendar for *days* days from *start_date*.
        Useful for "best day to travel" visualisations.
        """
        calendar = []
        for i in range(days):
            d = start_date + timedelta(days=i)
            price = self.predict_price(base_price_usd, d, category)
            calendar.append({
                "date": d.isoformat(),
                "day_of_week": d.strftime("%a"),
                "predicted_price_usd": price,
                "relative_price": price / base_price_usd,
            })
        return calendar

    def optimal_booking_window(
        self,
        base_price_usd: float,
        trip_date: date,
        category: PriceCategory,
        lookahead_days: int = 90,
    ) -> dict:
        """
        Simulate booking at different lead times and return the window
        with the lowest predicted price.
        """
        prices = {}
        for days_before in range(1, lookahead_days + 1):
            booking = trip_date - timedelta(days=days_before)
            if booking < date.today():
                continue
            prices[days_before] = self.predict_price(
                base_price_usd, trip_date, category, booking
            )

        if not prices:
            return {"message": "Trip date has passed"}

        best_lead = min(prices, key=prices.get)  # type: ignore
        return {
            "category": category.value,
            "trip_date": trip_date.isoformat(),
            "best_booking_days_before": best_lead,
            "best_booking_date": (trip_date - timedelta(days=best_lead)).isoformat(),
            "predicted_best_price_usd": prices[best_lead],
            "current_price_usd": self.predict_price(base_price_usd, trip_date, category),
            "savings_pct": round(
                (self.predict_price(base_price_usd, trip_date, category) - prices[best_lead])
                / self.predict_price(base_price_usd, trip_date, category) * 100, 1
            ),
        }

    def budget_forecast(
        self,
        flight_base: float,
        hotel_base_per_night: float,
        attraction_budget: float,
        trip_date: date,
        n_nights: int,
    ) -> dict:
        """Full trip budget forecast broken down by category."""
        flight    = self.predict_price(flight_base, trip_date, PriceCategory.FLIGHT)
        hotel     = self.predict_price(hotel_base_per_night, trip_date, PriceCategory.HOTEL) * n_nights
        transport = attraction_budget * 0.15  # rough 15% for local transport
        food      = n_nights * 60            # $60/day food estimate
        return {
            "flight_usd":      round(flight, 2),
            "hotel_usd":       round(hotel, 2),
            "attractions_usd": round(attraction_budget, 2),
            "food_usd":        round(food, 2),
            "transport_usd":   round(transport, 2),
            "total_usd":       round(flight + hotel + attraction_budget + food + transport, 2),
        }
