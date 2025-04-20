"""Map display utilities for map integration."""
import pandas as pd
import folium
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from streamlit_folium import folium_static

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
    
    zoom_level = 13
    if lat_diff > 0.1 or lon_diff > 0.1:
        zoom_level = 12
    if lat_diff > 0.2 or lon_diff > 0.2:
        zoom_level = 11
    if lat_diff > 0.5 or lon_diff > 0.5:
        zoom_level = 10
        
    return center_lat, center_lon, zoom_level

def add_points_to_map(m: folium.Map, df_points: pd.DataFrame) -> None:
    """Add points to the map.
    
    Args:
        m: Folium map object
        df_points: DataFrame with points data
    """
    for _, row in df_points.iterrows():
        popup_html = f"<b>{row.get('name', 'Точка')}</b>"
        
        # Добавляем рейтинг, если он есть
        if 'rating' in row and row['rating']:
            stars = '★' * int(row['rating']) + '☆' * (5 - int(row['rating']))
            popup_html += f"<br><span style='color: #FFD700;'>{stars}</span> {row['rating']}"
            
        # Добавляем количество отзывов, если они есть
        if 'reviews' in row and row['reviews']:
            popup_html += f"<br>Отзывы: {row['reviews']}"
            
        # Добавляем адрес, если он есть
        if 'address' in row and row['address']:
            popup_html += f"<br>Адрес: {row['address']}"
            
        # Добавляем телефон, если он есть
        if 'phone' in row and row['phone']:
            popup_html += f"<br>Телефон: {row['phone']}"
            
        # Создаем iframe для попапа с настраиваемым размером
        iframe = folium.IFrame(html=popup_html, width=300, height=150)
        popup = folium.Popup(iframe, max_width=300)
        
        # Цвет маркера зависит от рейтинга
        color = 'red'
        if 'rating' in row and row['rating']:
            if float(row['rating']) >= 4.5:
                color = 'green'
            elif float(row['rating']) >= 4.0:
                color = 'orange'
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=row.get('name', 'Точка')
        ).add_to(m)

def add_route_to_map(
    m: folium.Map,
    path_points: List[Dict[str, Any]],
    route_points: List[Dict[str, Any]]
) -> None:
    """Add route to the map.
    
    Args:
        m: Folium map object
        path_points: List of path points
        route_points: List of route points
    """
    # Add route line
    route_coords = [(p["lat"], p["lon"]) for p in path_points]
    folium.PolyLine(
        route_coords,
        color='blue',
        weight=5,
        opacity=0.7,
        tooltip='Маршрут'
    ).add_to(m)
    
    # Add start and end points
    for point in route_points:
        color = 'green' if point.get('is_start', False) else 'red'
        
        popup_html = f"<b>{point.get('name', 'Точка')}</b>"
        
        # Добавляем рейтинг, если он есть
        if 'rating' in point and point['rating']:
            stars = '★' * int(point['rating']) + '☆' * (5 - int(point['rating']))
            popup_html += f"<br><span style='color: #FFD700;'>{stars}</span> {point['rating']}"
            
        # Добавляем количество отзывов, если они есть
        if 'reviews' in point and point['reviews']:
            popup_html += f"<br>Отзывы: {point['reviews']}"
            
        # Добавляем адрес, если он есть
        if 'address' in point and point['address']:
            popup_html += f"<br>Адрес: {point['address']}"
        
        # Создаем iframe для попапа с настраиваемым размером
        iframe = folium.IFrame(html=popup_html, width=300, height=150)
        popup = folium.Popup(iframe, max_width=300)
        
        folium.CircleMarker(
            location=[point['lat'], point['lon']],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup,
            tooltip=point.get('name', 'Точка')
        ).add_to(m)

def display_map(
    pydeck_data: List[Dict[str, Any]],
    map_type: str = "points",
    path_points: Optional[List[Dict[str, Any]]] = None,
    route_points: Optional[List[Dict[str, Any]]] = None,
    title: str = "🗺️ Интерактивная карта"
) -> None:
    """Display map with points or route using Folium.
    
    Args:
        pydeck_data: List of points data
        map_type: Type of map ("points" or "route")
        path_points: List of path points for route type
        route_points: List of route points for route type
        title: Map title
    """
    if not pydeck_data and map_type == "points":
        return
        
    if map_type == "route" and (not path_points or not route_points):
        return
    
    with st.container():
        st.markdown("## ")
        st.subheader(title)
        st.markdown("---")
        
        if map_type == "points":
            df_points = pd.DataFrame(pydeck_data)
            center_lat, center_lon, zoom_level = configure_map_settings(df_points)
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, 
                          tiles="OpenStreetMap")
            add_points_to_map(m, df_points)
        else:
            df_route_points = pd.DataFrame(route_points)
            center_lat, center_lon, zoom_level = configure_map_settings(
                df_route_points, map_type, path_points
            )
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level,
                          tiles="OpenStreetMap")
            add_route_to_map(m, path_points, route_points)
        
        # Display the map in Streamlit
        folium_static(m) 