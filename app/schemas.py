from pydantic import BaseModel
from typing import List, Optional

class RouterData(BaseModel):
    ssid: str
    bssid: str
    signal_strength: int

class MeasurementData(BaseModel):
    room_name: str
    device_id: str
    timestamp: int
    routers: List[RouterData]

class PredictData(BaseModel):
    routers: List[RouterData]
    ignore_measurements: Optional[List[int]] = None
    use_remove_unreceived_bssids: Optional[bool] = True
    handle_missing_values_strategy: Optional[str] = "use_received"
    router_selection: Optional[str] = 'all'
    router_presence_threshold: Optional[float] = 0.0
    value_scaling_strategy: Optional[str] = 'none'
    router_rssi_threshold: Optional[int] = -100
    algorithm: Optional[str] = 'knn_euclidean'
    k_value: Optional[int] = 5
    weights: Optional[str] = 'uniform'
    n_estimators: Optional[int] = 300
    c_value: Optional[float] = 1.0
    gamma_value: Optional[float] = 1.0
    max_depth: Optional[int] = None
