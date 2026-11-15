from datetime import datetime
from typing import List, Optional, Dict, Any
from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import Field
from core.oxm.mongo.audit_base import AuditBase


class ClusterState(DocumentBase, AuditBase):
    """
    聚类状态文档模型
    
    每个群组维护一个聚类状态文档，存储该群组的完整聚类信息
    """
    
    # 主键
    group_id: Indexed(str) = Field(..., description="群组ID，主键")
    
    # 基础聚类信息
    event_ids: List[str] = Field(default_factory=list, description="所有 event_id 列表")
    timestamps: List[float] = Field(default_factory=list, description="时间戳列表")
    cluster_ids: List[str] = Field(default_factory=list, description="聚类 ID 列表")
    
    # event_id -> cluster_id 映射
    eventid_to_cluster: Dict[str, str] = Field(default_factory=dict, description="事件到聚类的映射")
    
    # 聚类元数据
    next_cluster_idx: int = Field(default=0, description="下一个聚类索引")
    
    # 聚类中心信息（向量存储为列表）
    cluster_centroids: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="聚类中心向量 {cluster_id: vector}"
    )
    cluster_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="聚类成员数量 {cluster_id: count}"
    )
    cluster_last_ts: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="聚类最后时间戳 {cluster_id: timestamp}"
    )
    
    class Settings:
        name = "cluster_states"

