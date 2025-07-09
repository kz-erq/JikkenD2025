# 6都市をコード内に記述する形で経路検索
# 衛星-地上間は最近の衛星を選択

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from RFrev import delay_sum
# --- 条件設定と準備 ---

# 衛星パラメータ
DEFAULT_NUM_ORBITS = 6  # 軌道数
DEFAULT_NUM_SATS_PER_ORBIT = 11  # 軌道当たりの衛星数
DEFAULT_INCLINATION_DEG = 80.0  # 軌道傾斜角 (度)

# 地球パラメータ
EARTH_RADIUS_KM = 6371.0  # 地球の半径 (km)
# 衛星高度 (LEO: 低軌道衛星を想定)
SATELLITE_ALTITUDE_KM = 1000.0  # 衛星の高度 (km)
SATELLITE_ORBIT_RADIUS_KM = EARTH_RADIUS_KM + SATELLITE_ALTITUDE_KM  # 衛星の軌道半径 (km)

# 光速 (km/s)
SPEED_OF_LIGHT_KM_S = 299792.458

# 地上局の初期設定 (緯度, 経度) - 英語名、シドニーとケープタウンを追加
DEFAULT_GROUND_STATIONS_GEO = {
    "Tokyo": [35.6895, 139.6917],
    "WashingtonDC": [38.9072, -77.0369], # ワシントンD.C.
    "London": [51.5074, -0.1278],
    "NewDelhi": [28.6139, 77.2090],     # ニューデリー
    "Sydney": [-33.8688, 151.2093],     # シドニー
    "CapeTown": [-33.9249, 18.4241],   # ケープタウン
}

# --- 関数定義 ---

def geodetic_to_ecef(lat_deg, lon_deg, alt_km, earth_radius_km=EARTH_RADIUS_KM):
    """
    地理座標 (緯度、経度、高度) を地心地球固定 (ECEF) 座標に変換する。

    Args:
        lat_deg (float): 緯度 (度)。
        lon_deg (float): 経度 (度)。
        alt_km (float): 高度 (km)。
        earth_radius_km (float, optional): 地球の半径 (km)。デフォルトは EARTH_RADIUS_KM。

    Returns:
        numpy.ndarray: ECEF座標 [x, y, z] (km)。
    """
    lat_rad = np.deg2rad(lat_deg)  # 緯度をラジアンに変換
    lon_rad = np.deg2rad(lon_deg)  # 経度をラジアンに変換
    
    # 簡単のため球体地球を仮定
    x = (earth_radius_km + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (earth_radius_km + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (earth_radius_km + alt_km) * np.sin(lat_rad)
    return np.array([x, y, z])

def get_satellite_data(num_orbits, num_sats_per_orbit, satellite_orbit_radius_km, inclination_deg):
    """
    衛星のECEF座標とIDマッピングを生成する。

    Args:
        num_orbits (int): 軌道数。
        num_sats_per_orbit (int): 軌道当たりの衛星数。
        satellite_orbit_radius_km (float): 衛星の軌道半径 (km)。
        inclination_deg (float): 軌道傾斜角 (度)。

    Returns:
        tuple:
            - sats_ecef_coords (dict): 衛星のECEF座標 {global_sat_id: np.array([x,y,z])}。
            - global_id_to_orbital_indices (dict): グローバルIDから軌道インデックスへのマッピング {global_sat_id: (orbit_idx, sat_idx_in_orbit)}。
            - orbital_indices_to_global_id (dict): 軌道インデックスからグローバルIDへのマッピング {(orbit_idx, sat_idx_in_orbit): global_sat_id}。
    """
    inclination_rad = np.deg2rad(inclination_deg)  # 軌道傾斜角をラジアンに変換
    
    sats_ecef_coords = {}  # 衛星IDをキー、ECEF座標を値とする辞書
    global_id_to_orbital_indices = {} # グローバルIDをキー、(軌道インデックス, 軌道内衛星インデックス)を値とする辞書
    orbital_indices_to_global_id = {} # (軌道インデックス, 軌道内衛星インデックス)をキー、グローバルIDを値とする辞書
    
    global_sat_id_counter = 0  # 衛星に割り当てるグローバルIDのカウンタ
    for i in range(num_orbits):  # 各軌道についてループ
        raan_rad = np.deg2rad(360.0 * i / num_orbits)  # 赤道昇交点経度 (RAAN)
        
        for k in range(num_sats_per_orbit):  # 各軌道内の衛星についてループ
            # 隣接する軌道間で衛星の位相をずらすためのオフセット (Walker配置のような効果)
            phase_offset_deg = (360.0 / (2.0 * num_sats_per_orbit)) * (i % 2) if num_orbits > 1 else 0.0
            true_anomaly_rad = np.deg2rad((360.0 * k / num_sats_per_orbit) + phase_offset_deg)  # 真近点角 (軌道面内での位置)
            
            # 軌道面内での座標 (z_orb = 0)
            x_orb = satellite_orbit_radius_km * np.cos(true_anomaly_rad)
            y_orb = satellite_orbit_radius_km * np.sin(true_anomaly_rad)
            
            # 軌道傾斜角による回転
            x_rot_inc = x_orb
            y_rot_inc = y_orb * np.cos(inclination_rad)
            z_rot_inc = y_orb * np.sin(inclination_rad)
            
            # RAANによる回転
            x_ecef = x_rot_inc * np.cos(raan_rad) - y_rot_inc * np.sin(raan_rad)
            y_ecef = x_rot_inc * np.sin(raan_rad) + y_rot_inc * np.cos(raan_rad)
            z_ecef = z_rot_inc
            
            sats_ecef_coords[global_sat_id_counter] = np.array([x_ecef, y_ecef, z_ecef])
            global_id_to_orbital_indices[global_sat_id_counter] = (i, k)
            orbital_indices_to_global_id[(i, k)] = global_sat_id_counter
            global_sat_id_counter += 1
            
    return sats_ecef_coords, global_id_to_orbital_indices, orbital_indices_to_global_id

def create_satellite_network_graph(sats_ecef_coords, global_id_to_orbital_indices, orbital_indices_to_global_id, num_orbits, num_sats_per_orbit):
    """
    衛星ネットワークのグラフを生成する。
    ノードは衛星、エッジは衛星間リンク(ISL)。エッジの重みは衛星間の直線距離。

    Args:
        sats_ecef_coords (dict): 衛星のECEF座標。
        global_id_to_orbital_indices (dict): グローバルIDから軌道インデックスへのマッピング。
        orbital_indices_to_global_id (dict): 軌道インデックスからグローバルIDへのマッピング。
        num_orbits (int): 軌道数。
        num_sats_per_orbit (int): 軌道当たりの衛星数。

    Returns:
        networkx.Graph: 衛星ネットワークを表すグラフ。
    """
    graph = nx.Graph()  # 無向グラフを作成
    # 全ての衛星をノードとしてグラフに追加
    for sat_id in sats_ecef_coords.keys():
        graph.add_node(sat_id)
        
    # ISL (衛星間リンク) をエッジとしてグラフに追加
    for current_global_id, (orb_idx, sat_idx_in_orbit) in global_id_to_orbital_indices.items():
        current_pos = sats_ecef_coords[current_global_id]  # 現在の衛星の座標
        
        # 1. 同一軌道上の衛星とのリンク (前後)
        next_sat_idx_in_orbit = (sat_idx_in_orbit + 1) % num_sats_per_orbit  # 同じ軌道上の次の衛星のインデックス
        next_sat_global_id_intra = orbital_indices_to_global_id[(orb_idx, next_sat_idx_in_orbit)]
        next_pos_intra = sats_ecef_coords[next_sat_global_id_intra]
        dist_intra = np.linalg.norm(current_pos - next_pos_intra)  # 衛星間距離
        graph.add_edge(current_global_id, next_sat_global_id_intra, weight=dist_intra) # エッジを追加 (重みは距離)
        
        # 2. 隣接軌道の衛星とのリンク (軌道数が1より大きい場合のみ)
        if num_orbits > 1:
            next_orb_idx = (orb_idx + 1) % num_orbits  # 隣の軌道のインデックス (リング状)
            # 隣接軌道の同じ軌道内インデックスを持つ衛星と接続 (位相オフセットは座標生成時に考慮済み)
            next_sat_global_id_inter = orbital_indices_to_global_id[(next_orb_idx, sat_idx_in_orbit)]
            next_pos_inter = sats_ecef_coords[next_sat_global_id_inter]
            dist_inter = np.linalg.norm(current_pos - next_pos_inter)
            graph.add_edge(current_global_id, next_sat_global_id_inter, weight=dist_inter)
            
    return graph

def find_closest_satellite(gs_ecef_coord, sats_ecef_coords):
    """
    指定された地上局のECEF座標に最も近い衛星とその距離を見つける。

    Args:
        gs_ecef_coord (numpy.ndarray): 地上局のECEF座標。
        sats_ecef_coords (dict): 全衛星のECEF座標の辞書。

    Returns:
        tuple:
            - closest_sat_id (int): 最も近い衛星のグローバルID。
            - min_dist (float): 地上局と最も近い衛星との距離 (km)。
    """
    min_dist = float('inf')  # 最小距離を無限大で初期化
    closest_sat_id = -1      # 最も近い衛星のIDを初期化
    for sat_id, sat_pos in sats_ecef_coords.items():
        dist = np.linalg.norm(gs_ecef_coord - sat_pos)  # 地上局と衛星間の距離を計算
        if dist < min_dist:
            min_dist = dist
            closest_sat_id = sat_id
    return closest_sat_id, min_dist



def plot_constellation_and_path(
    sats_ecef_coords,
    sat_network_graph,
    ground_stations_ecef,
    gs1_name, gs1_ecef,
    gs2_name, gs2_ecef,
    uplink_sat_id,
    downlink_sat_id,
    path_sat_ids,
    earth_radius_km=EARTH_RADIUS_KM
):
    """
    地球、衛星、ISL、地上局、および指定された経路を3Dプロットする。

    Args:
        sats_ecef_coords (dict): 衛星のECEF座標。
        sat_network_graph (networkx.Graph): 衛星ネットワークグラフ。
        ground_stations_ecef (dict): 地上局のECEF座標。
        gs1_name (str): 地上局1の名前。
        gs1_ecef (numpy.ndarray): 地上局1のECEF座標。
        gs2_name (str): 地上局2の名前。
        gs2_ecef (numpy.ndarray): 地上局2のECEF座標。
        uplink_sat_id (int): アップリンク衛星のID。
        downlink_sat_id (int): ダウンリンク衛星のID。
        path_sat_ids (list): 経路上の衛星IDのリスト。
        earth_radius_km (float, optional): 地球の半径 (km)。
    """
    fig = plt.figure(figsize=(14, 14)) # プロット領域のサイズ指定
    ax = fig.add_subplot(111, projection='3d') # 3Dプロット用のサブプロットを追加

    # 地球の描画
    u_earth = np.linspace(0, 2 * np.pi, 100)
    v_earth = np.linspace(0, np.pi, 50)
    x_earth = earth_radius_km * np.outer(np.cos(u_earth), np.sin(v_earth))
    y_earth = earth_radius_km * np.outer(np.sin(u_earth), np.sin(v_earth))
    z_earth = earth_radius_km * np.outer(np.ones(np.size(u_earth)), np.cos(v_earth))
    ax.plot_surface(x_earth, y_earth, z_earth, color='deepskyblue', alpha=0.3, rstride=4, cstride=4, linewidth=0, label='Earth')

    # 衛星の描画
    sat_x = [pos[0] for pos in sats_ecef_coords.values()]
    sat_y = [pos[1] for pos in sats_ecef_coords.values()]
    sat_z = [pos[2] for pos in sats_ecef_coords.values()]
    ax.scatter(sat_x, sat_y, sat_z, color='orangered', s=12, label='Satellites', alpha=0.8)

    # ISLの描画 (凡例が重複しないように)
    isl_label_added = False
    for u_node, v_node in sat_network_graph.edges():
        pos_u = sats_ecef_coords[u_node]
        pos_v = sats_ecef_coords[v_node]
        if not isl_label_added:
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]], 'gray', alpha=0.3, linewidth=2.0, label='ISLs')
            isl_label_added = True
        else:
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]], 'gray', alpha=0.3, linewidth=2.0)

    # 地上局の描画 (凡例が重複しないように)
    gs_label_added = False
    for name, pos in ground_stations_ecef.items():
        label_val = 'Ground Stations' if not gs_label_added else "" # 最初の地上局にのみラベルを付与
        ax.scatter(pos[0], pos[1], pos[2], color='forestgreen', s=60, marker='^', label=label_val, edgecolors='black', linewidth=0.5)
        ax.text(pos[0]*1.03, pos[1]*1.03, pos[2]*1.03, name, color='black', fontsize=9) # 地上局名を表示
        gs_label_added = True
        
    # 選択された経路のハイライト
    uplink_sat_pos = sats_ecef_coords[uplink_sat_id]
    downlink_sat_pos = sats_ecef_coords[downlink_sat_id]

    # アップリンク部分の描画
    ax.plot([gs1_ecef[0], uplink_sat_pos[0]], [gs1_ecef[1], uplink_sat_pos[1]], [gs1_ecef[2], uplink_sat_pos[2]], 
            'mediumorchid', linewidth=2.5, label=f'Path: {gs1_name} to {gs2_name}') # 経路全体のラベル
    
    # 衛星間経路部分の描画
    if path_sat_ids and len(path_sat_ids) > 1: # ISLが存在する場合
        for i in range(len(path_sat_ids) - 1):
            sat1_pos = sats_ecef_coords[path_sat_ids[i]]
            sat2_pos = sats_ecef_coords[path_sat_ids[i+1]]
            ax.plot([sat1_pos[0], sat2_pos[0]], [sat1_pos[1], sat2_pos[1]], [sat1_pos[2], sat2_pos[2]], 
                    'mediumorchid', linewidth=2.5) # 同じ色で描画

    # ダウンリンク部分の描画
    ax.plot([downlink_sat_pos[0], gs2_ecef[0]], [downlink_sat_pos[1], gs2_ecef[1]], [downlink_sat_pos[2], gs2_ecef[2]], 
            'mediumorchid', linewidth=2.5) # 同じ色で描画

    # 経路上の衛星を強調表示
    path_sat_coords_x = [sats_ecef_coords[sat_id][0] for sat_id in path_sat_ids]
    path_sat_coords_y = [sats_ecef_coords[sat_id][1] for sat_id in path_sat_ids]
    path_sat_coords_z = [sats_ecef_coords[sat_id][2] for sat_id in path_sat_ids]
    ax.scatter(path_sat_coords_x, path_sat_coords_y, path_sat_coords_z, color='gold', s=40, edgecolor='black', zorder=10, label='Path Satellites')

    ax.set_xlabel("X (km)")  # X軸ラベル
    ax.set_ylabel("Y (km)")  # Y軸ラベル
    ax.set_zlabel("Z (km)")  # Z軸ラベル
    ax.set_title(f"Satellite Network: {gs1_name} to {gs2_name}", fontsize=16) # グラフタイトル
    
    # 軸の範囲を調整し、アスペクト比を等しくする
    max_val = SATELLITE_ORBIT_RADIUS_KM * 1.1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_box_aspect([1,1,1]) # アスペクト比を1:1:1に設定

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.) # 凡例をグラフの外に表示
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 凡例のためのスペースを確保
    plt.show() # プロットを表示

# --- メイン処理 ---
def main():
    print("🛰️ Satellite Network Simulation 🛰️") # シミュレーション開始メッセージ
    print("-" * 30)

    # パラメータ設定
    num_orbits = DEFAULT_NUM_ORBITS
    num_sats_per_orbit = DEFAULT_NUM_SATS_PER_ORBIT
    inclination_deg = DEFAULT_INCLINATION_DEG
    ground_stations_geo = DEFAULT_GROUND_STATIONS_GEO # 地上局リストを使用

    print(f"Settings:") # 設定値を表示
    print(f"  Number of orbits: {num_orbits}")
    print(f"  Satellites per orbit: {num_sats_per_orbit}")
    print(f"  Inclination: {inclination_deg}°")
    print(f"  Satellite altitude: {SATELLITE_ALTITUDE_KM} km")
    print("-" * 30)

    # 1. 衛星の座標とIDマッピングを計算
    sats_ecef_coords, global_id_to_orbital_indices, orbital_indices_to_global_id = \
        get_satellite_data(num_orbits, num_sats_per_orbit, SATELLITE_ORBIT_RADIUS_KM, inclination_deg)
    
    print(f"Total number of satellites: {len(sats_ecef_coords)}") # 総衛星数を表示

    # 2. 地上局のECEF座標を計算
    ground_stations_ecef = {
        name: geodetic_to_ecef(lat, lon, 0, EARTH_RADIUS_KM) 
        for name, (lat, lon) in ground_stations_geo.items()
    }

    # 3. 衛星ネットワークグラフを生成
    sat_network_graph = create_satellite_network_graph(
        sats_ecef_coords, global_id_to_orbital_indices, orbital_indices_to_global_id,
        num_orbits, num_sats_per_orbit
    )
    print(f"Satellite network graph created (Nodes: {sat_network_graph.number_of_nodes()}, Edges: {sat_network_graph.number_of_edges()})")
    print("-" * 30)

    # 4. 地上局の選択
    available_gs_names = list(ground_stations_geo.keys()) # 利用可能な地上局名のリスト
    print(f"Available ground stations: {', '.join(available_gs_names)}") # 地上局名を表示
    
    while True: # 地上局1の入力ループ
        gs1_name = input(f"Select Ground Station 1 (e.g., Tokyo): ").strip()
        if gs1_name in available_gs_names:
            break
        print(f"Error: '{gs1_name}' is an invalid ground station name. Please choose from the list above.") # エラーメッセージ
        
    while True: # 地上局2の入力ループ
        gs2_name = input(f"Select Ground Station 2 (e.g., London): ").strip()
        if gs2_name in available_gs_names:
            if gs2_name != gs1_name: # 地上局1と2が異なることを確認
                break
            else:
                print("Error: Ground Station 1 and Ground Station 2 must be different.") # エラーメッセージ
        else:
            print(f"Error: '{gs2_name}' is an invalid ground station name. Please choose from the list above.") # エラーメッセージ
            
    gs1_ecef = ground_stations_ecef[gs1_name] # 地上局1のECEF座標を取得
    gs2_ecef = ground_stations_ecef[gs2_name] # 地上局2のECEF座標を取得
    
    print("-" * 30)
    print(f"Pathfinding: {gs1_name} ➡️ {gs2_name}") # 経路探索の開始を表示

    # 5. 各地上局に最も近い衛星を見つける
    uplink_sat_id, dist_uplink_km = find_closest_satellite(gs1_ecef, sats_ecef_coords)
    downlink_sat_id, dist_downlink_km = find_closest_satellite(gs2_ecef, sats_ecef_coords)
    
    # 最寄り衛星のECEF座標を取得
    uplink_sat_ecef = sats_ecef_coords[uplink_sat_id]
    downlink_sat_ecef = sats_ecef_coords[downlink_sat_id]

    print(f"  Closest satellite to {gs1_name}: ID {uplink_sat_id} (Orbit {global_id_to_orbital_indices[uplink_sat_id][0]}, Sat {global_id_to_orbital_indices[uplink_sat_id][1]}), Distance: {dist_uplink_km:.2f} km")
    print(f"  Closest satellite to {gs2_name}: ID {downlink_sat_id} (Orbit {global_id_to_orbital_indices[downlink_sat_id][0]}, Sat {global_id_to_orbital_indices[downlink_sat_id][1]}), Distance: {dist_downlink_km:.2f} km")

    # 6. 衛星間リンクの最短経路を探索し、総伝搬距離を計算
    # total_propagation_distance_km は表示用のため、dist_uplink_km と dist_downlink_km を使用して初期化
    total_propagation_distance_km = dist_uplink_km + dist_downlink_km
    dist_isl_total_km = 0.0 # ISL区間の合計距離を初期化
    
    if uplink_sat_id == downlink_sat_id: # アップリンク衛星とダウンリンク衛星が同じ場合
        path_sat_ids = [uplink_sat_id] # 経路はアップリンク衛星のみ
        print(f"  Uplink and downlink satellites are the same (ID {uplink_sat_id}). No ISL communication.")
    else: # アップリンク衛星とダウンリンク衛星が異なる場合
        try:
            # NetworkX を使用して重み付きグラフで最短経路を探索
            path_sat_ids = nx.shortest_path(sat_network_graph, source=uplink_sat_id, target=downlink_sat_id, weight='weight')
            dist_isl_total_km = nx.shortest_path_length(sat_network_graph, source=uplink_sat_id, target=downlink_sat_id, weight='weight')
            total_propagation_distance_km += dist_isl_total_km # ISLの距離を総距離に加算
            print(f"  Shortest ISL path total distance: {dist_isl_total_km:.2f} km")
        except nx.NetworkXNoPath: # 経路が見つからない場合のエラーハンドリング
            print(f"Error: No path found between satellite {uplink_sat_id} and satellite {downlink_sat_id}.")
            return
        except nx.NodeNotFound: # ノード (衛星ID) がグラフ内に存在しない場合のエラーハンドリング
            print(f"Error: Satellite ID {uplink_sat_id} or {downlink_sat_id} not found in the graph.")
            return
            
    print("-" * 30)
    print("--- Communication Path and Link Delay Details ---") # 通信経路と遅延詳細の開始

    # 7. 各リンクの遅延を計算して出力
    # 7-1. アップリンク遅延 (delay_sum 関数を使用)
    # delay_sum(pos1[km], pos2[km], data_size[MB], rain_rate[mm/h], cloud_density[g/m^3], freq[GHz]): # For Satellite-to-Gateway
    uplink_delay_ms = delay_sum(gs1_ecef, uplink_sat_ecef, 1, 10, 0.5, 20) * 1000 # ミリ秒に変換
    orb_idx_up, sat_in_orb_idx_up = global_id_to_orbital_indices[uplink_sat_id]
    print(f"1. Uplink: Ground Station {gs1_name} ➡️ Satellite ID {uplink_sat_id} ([Orbit{orb_idx_up}, Sat{sat_in_orb_idx_up}])")
    print(f"   - Distance: {dist_uplink_km:.2f} km") # 距離は find_closest_satellite の結果を使用
    print(f"   - Delay: {uplink_delay_ms:.2f} ms")

    # 7-2. 衛星間リンク (ISL) 遅延 (従来通り計算)
    total_isl_delay_ms = 0 # ISL合計遅延を初期化
    if uplink_sat_id != downlink_sat_id and len(path_sat_ids) > 1 : # ISLが存在する場合
        print(f"2. Inter-Satellite Links (ISL):")
        for i in range(len(path_sat_ids) - 1): # 経路上の各ISL区間についてループ
            current_hop_sat_id = path_sat_ids[i]
            next_hop_sat_id = path_sat_ids[i+1]
            
            orb_idx_curr, sat_in_orb_idx_curr = global_id_to_orbital_indices[current_hop_sat_id]
            orb_idx_next, sat_in_orb_idx_next = global_id_to_orbital_indices[next_hop_sat_id]
            
            dist_leg_km = sat_network_graph[current_hop_sat_id][next_hop_sat_id]['weight'] # ISL区間の距離
            isl_leg_delay_ms = (dist_leg_km / SPEED_OF_LIGHT_KM_S) * 1000 # ISL区間の遅延 (ミリ秒)
            total_isl_delay_ms += isl_leg_delay_ms # ISL合計遅延に加算
            
            print(f"   Satellite ID {current_hop_sat_id} ([Orbit{orb_idx_curr}, Sat{sat_in_orb_idx_curr}]) ➡️ Satellite ID {next_hop_sat_id} ([Orbit{orb_idx_next}, Sat{sat_in_orb_idx_next}])")
            print(f"     - Distance: {dist_leg_km:.2f} km")
            print(f"     - Delay: {isl_leg_delay_ms:.2f} ms")
    else: # ISLが存在しない場合
        print(f"2. Inter-Satellite Links (ISL): None")

    # 7-3. ダウンリンク遅延 (delay_sum 関数を使用)
    downlink_delay_ms = delay_sum(gs2_ecef, downlink_sat_ecef, 1, 10, 0.5, 20) * 1000 # ミリ秒に変換
    orb_idx_down, sat_in_orb_idx_down = global_id_to_orbital_indices[downlink_sat_id]
    print(f"3. Downlink: Satellite ID {downlink_sat_id} ([Orbit{orb_idx_down}, Sat{sat_in_orb_idx_down}]) ➡️ Ground Station {gs2_name}")
    print(f"   - Distance: {dist_downlink_km:.2f} km") # 距離は find_closest_satellite の結果を使用
    print(f"   - Delay: {downlink_delay_ms:.2f} ms")
    
    print("-" * 30)
    print("--- Totals ---") # 合計結果の表示
    total_propagation_time_ms = uplink_delay_ms + total_isl_delay_ms + downlink_delay_ms # 総伝搬時間
    print(f"📡 Total Propagation Distance: {total_propagation_distance_km:.2f} km") # 総伝搬距離
    print(f"⏱️ Total Propagation Time: {total_propagation_time_ms:.2f} ms") # 総伝搬時間
    # 総伝搬時間の内訳を表示
    print(f"   (Breakdown: Uplink {uplink_delay_ms:.2f} ms + ISL Total {total_isl_delay_ms:.2f} ms + Downlink {downlink_delay_ms:.2f} ms)")
    print("-" * 30)

    # 8. 3Dプロット
    print("Generating 3D plot...") # プロット生成中メッセージ
    plot_constellation_and_path(
        sats_ecef_coords,
        sat_network_graph,
        ground_stations_ecef,
        gs1_name, gs1_ecef,
        gs2_name, gs2_ecef,
        uplink_sat_id,
        downlink_sat_id,
        path_sat_ids,
        EARTH_RADIUS_KM
    )

if __name__ == "__main__":
    main()