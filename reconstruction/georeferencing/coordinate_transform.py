"""Coordinate Transformation Module Using pyproj.

Transforms coordinates between geographic WGS84 (EPSG:4326) and projected metric
systems (UTM or local projections). Automatically computes UTM zones, guarantees
RFC 7946 [longitude, latitude, altitude] ordering for GeoJSON, and preserves CRS metadata.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import pyproj

logger = logging.getLogger("georeferencing.transform")


def determine_utm_crs(longitude: float, latitude: float) -> str:
    """Automatically determines the standard UTM EPSG code from longitude and latitude."""
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"Longitude out of valid bounds [-180, 180]: {longitude}")
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"Latitude out of valid bounds [-90, 90]: {latitude}")

    zone = int(math.floor((longitude + 180.0) / 6.0)) + 1
    if zone > 60:
        zone = 60
    if zone < 1:
        zone = 1

    epsg = 32600 + zone if latitude >= 0.0 else 32700 + zone
    return f"EPSG:{epsg}"


class CoordinateTransformer:
    """Bidirectional coordinate transformer between WGS84 and projected metric systems."""

    def __init__(
        self,
        source_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        reference_point: Optional[Tuple[float, float]] = None,
        altitude_reference: str = "WGS84_ellipsoidal",
    ) -> None:
        self.source_crs = source_crs
        self.altitude_reference = altitude_reference

        if target_crs is None:
            if reference_point is None:
                raise ValueError("Either target_crs or reference_point (lon, lat) must be provided")
            ref_lon, ref_lat = reference_point
            self.target_crs = determine_utm_crs(ref_lon, ref_lat)
        else:
            self.target_crs = target_crs

        # Use always_xy=True to guarantee (x, y) = (lon, lat) or (easting, northing)
        self.forward_proj = pyproj.Transformer.from_crs(
            self.source_crs,
            self.target_crs,
            always_xy=True,
        )
        self.inverse_proj = pyproj.Transformer.from_crs(
            self.target_crs,
            self.source_crs,
            always_xy=True,
        )

        # Inspect target CRS name
        try:
            crs_obj = pyproj.CRS.from_user_input(self.target_crs)
            self.coordinate_system_name = crs_obj.name
        except Exception:
            self.coordinate_system_name = self.target_crs

    def wgs84_to_metric(self, lon: float, lat: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """Transforms WGS84 (lon, lat, alt) into target projected metric coordinates (x, y, z)."""
        x, y, z = self.forward_proj.transform(lon, lat, alt)
        return float(x), float(y), float(z)

    def metric_to_wgs84(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Transforms projected metric coordinates (x, y, z) into WGS84 (lon, lat, alt)."""
        lon, lat, alt = self.inverse_proj.transform(x, y, z)
        return float(lon), float(lat), float(alt)

    def batch_wgs84_to_metric(
        self,
        coords: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """Transforms a batch of (lon, lat, alt) coordinates into metric (x, y, z)."""
        return [self.wgs84_to_metric(lon, lat, alt) for lon, lat, alt in coords]

    def batch_metric_to_wgs84(
        self,
        coords: List[Tuple[float, float, float]],
    ) -> List[Tuple[float, float, float]]:
        """Transforms a batch of metric (x, y, z) coordinates into WGS84 (lon, lat, alt)."""
        return [self.metric_to_wgs84(x, y, z) for x, y, z in coords]

    def format_geojson_point(self, lon: float, lat: float, alt: float = 0.0) -> List[float]:
        """Formats coordinates strictly in RFC 7946 order: [longitude, latitude, altitude]."""
        return [round(float(lon), 7), round(float(lat), 7), round(float(alt), 3)]

    def get_crs_metadata(self) -> Dict[str, Any]:
        """Returns standard CRS metadata dictionary."""
        return {
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "coordinate_system": self.coordinate_system_name,
            "altitude_reference": self.altitude_reference,
            "geojson_coordinate_order": "[longitude, latitude, altitude] (RFC 7946)",
        }
