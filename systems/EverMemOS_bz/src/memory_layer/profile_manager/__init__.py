"""Profile Manager - Automatic profile extraction integrated with clustering.

This module provides ProfileManager, a core component that automatically extracts
and maintains user profiles based on clustered conversations.

Key Features:
- Automatic profile extraction triggered by cluster updates
- Incremental profile merging with version history
- Value discrimination to filter high-quality updates
- Seamless integration with ConvMemCellExtractor

Usage:
    from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
    
    # Initialize
    config = ProfileManagerConfig(
        scenario="group_chat",  # or "assistant"
        min_confidence=0.6,
        enable_versioning=True
    )
    profile_mgr = ProfileManager(llm_provider, config)
    
    # Attach to memcell extractor
    memcell_extractor = ConvMemCellExtractor(llm_provider)
    profile_mgr.attach_to_extractor(memcell_extractor)
    
    # Profiles are now automatically extracted and updated!
    # Access latest profiles:
    user_profile = profile_mgr.get_profile(user_id)
    all_profiles = profile_mgr.get_all_profiles()
"""

from memory_layer.profile_manager.config import ProfileManagerConfig, ScenarioType
from memory_layer.profile_manager.manager import ProfileManager
from memory_layer.profile_manager.discriminator import (
    ValueDiscriminator,
    DiscriminatorConfig,
)
from memory_layer.profile_manager.storage import (
    ProfileStorage,
    InMemoryProfileStorage,
)
from memory_layer.profile_manager.mongo_profile_storage import MongoProfileStorage

__all__ = [
    "ProfileManager",
    "ProfileManagerConfig",
    "ScenarioType",
    "ValueDiscriminator",
    "DiscriminatorConfig",
    "ProfileStorage",
    "InMemoryProfileStorage",
    "MongoProfileStorage",
]

