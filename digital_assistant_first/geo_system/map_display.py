"""Map display utilities for 2GIS integration."""
import pandas as pd
import pydeck as pdk
from typing import List, Dict, Any, Optional, Tuple

def configure_map_settings(
    df_points: pd.DataFrame,
    map_type: str = "points",
    path_points: Optional[List[Dict[str, Any]]] = None
) -> Tuple[float, float, int]:
    """Configure map view settings based on data.
    
    Args:
        df_points: DataFrame with points data
        map_type: Type of map ("points" or "route")
        path_points: List of path points for route type
        
    Returns:
        Tuple of (center_lat, center_lon, zoom_level)
    """
    center_lat = df_points["lat"].mean()
    center_lon = df_points["lon"].mean()
    
    if map_type == "route" and path_points:
        max_lat = max(p["lat"] for p in path_points)
        min_lat = min(p["lat"] for p in path_points)
        max_lon = max(p["lon"] for p in path_points)
        min_lon = min(p["lon"] for p in path_points)
    else:
        max_lat = df_points["lat"].max()
        min_lat = df_points["lat"].min()
        max_lon = df_points["lon"].max()
        min_lon = df_points["lon"].min()
    
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    
    zoom_level = 10
    if lat_diff > 0.1 or lon_diff > 0.1:
        zoom_level = 9
    if lat_diff > 0.2 or lon_diff > 0.2:
        zoom_level = 8
    if lat_diff > 0.5 or lon_diff > 0.5:
        zoom_level = 7
        
    return center_lat, center_lon, zoom_level

def create_points_layer(df_points: pd.DataFrame) -> pdk.Layer:
    """Create a ScatterplotLayer for points display.
    
    Args:
        df_points: DataFrame with points data
        
    Returns:
        pydeck Layer for points display
    """
    return pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position="[lon, lat]",
        get_radius=30,
        radiusMinPixels=6,
        radiusMaxPixels=100,
        radiusScale=0.8,
        get_fill_color=[255, 0, 0],
        pickable=True,
    )

def create_route_layer(
    path_points: List[Dict[str, Any]],
    route_points: List[Dict[str, Any]]
) -> List[pdk.Layer]:
    """Create layers for route display.
    
    Args:
        path_points: List of path points
        route_points: List of route points
        
    Returns:
        List of pydeck Layers for route display
    """
    path_data = [{
        "path": [[p["lon"], p["lat"]] for p in path_points],
        "name": "Маршрут"
    }]
    
    df_route_points = pd.DataFrame(route_points)
    
    return [
        pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_width=5,
            get_color=[0, 0, 255],
            width_min_pixels=3,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=df_route_points,
            get_position="[lon, lat]",
            get_radius=50,
            radiusMinPixels=8,
            radiusMaxPixels=100,
            radiusScale=1,
            get_fill_color=["is_start ? 0 : 255", "is_start ? 200 : 0", "is_start ? 0 : 0", 200],
            pickable=True,
        )
    ]

def display_2gis_map(
    pydeck_data: List[Dict[str, Any]],
    map_type: str = "points",
    path_points: Optional[List[Dict[str, Any]]] = None,
    route_points: Optional[List[Dict[str, Any]]] = None,
    title: str = "🗺️ Интерактивная карта 2GIS"
) -> None:
    """Display 2GIS map with points or route.
    
    Args:
        pydeck_data: List of points data
        map_type: Type of map ("points" or "route")
        path_points: List of path points for route type
        route_points: List of route points for route type
        title: Map title
    """
    import streamlit as st
    
    if not pydeck_data and map_type == "points":
        return
        
    if map_type == "route" and (not path_points or not route_points):
        return
    
    with st.container():
        st.markdown("## ")
        st.subheader(title)
        st.markdown("---")
        
        if map_type == "points":
            df_pydeck = pd.DataFrame(pydeck_data)
            center_lat, center_lon, zoom_level = configure_map_settings(df_pydeck)
            layers = [create_points_layer(df_pydeck)]
        else:
            df_route_points = pd.DataFrame(route_points)
            center_lat, center_lon, zoom_level = configure_map_settings(
                df_route_points, map_type, path_points
            )
            layers = create_route_layer(path_points, route_points)
        
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=zoom_level,
                ),
                layers=layers,
                tooltip={
                    "html": "<b>{name}</b>",
                    "style": {"color": "white"},
                },
            )
        ) 