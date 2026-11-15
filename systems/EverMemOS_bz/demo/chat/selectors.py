"""交互式选择器

提供语言、场景、群组的选择功能。
"""

from typing import List, Dict, Any, Optional

from demo.config import ScenarioType
from demo.utils import query_all_groups_from_mongodb
from demo.ui import I18nTexts
from common_utils.cli_ui import CLIUI


class LanguageSelector:
    """语言选择器"""
    
    @staticmethod
    def select_language() -> str:
        """交互式选择语言
        
        Returns:
            语言代码："zh" 或 "en"
        """
        print()
        print("=" * 60)
        print("  🌏  语言选择 / Language Selection")
        print("=" * 60)
        print()
        print("  [1] 中文 (Chinese)")
        print("  [2] English")
        print()
        
        while True:
            try:
                choice = input("请选择语言 / Please select language [1-2]: ").strip()
                if not choice:
                    continue
                
                index = int(choice)
                if index == 1:
                    return "zh"
                elif index == 2:
                    return "en"
                else:
                    print("❌ 请输入 1 或 2 / Please enter 1 or 2\n")
            
            except ValueError:
                print("❌ 请输入有效的数字 / Please enter a valid number\n")
            except KeyboardInterrupt:
                print("\n")
                return "zh"


class ScenarioSelector:
    """场景模式选择器"""
    
    @staticmethod
    def select_scenario(texts: I18nTexts) -> Optional[ScenarioType]:
        """交互式选择场景模式
        
        Args:
            texts: 国际化文本对象
            
        Returns:
            ScenarioType 或 None（取消）
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("scenario_selection_title"))
        print()
        
        print(f"  [1] {texts.get('scenario_assistant')}")
        print(f"      {texts.get('scenario_assistant_desc')}")
        print()
        print(f"  [2] {texts.get('scenario_group_chat')}")
        print(f"      {texts.get('scenario_group_chat_desc')}")
        print()
        
        while True:
            try:
                choice = input(f"{texts.get('scenario_prompt')}: ").strip()
                if not choice:
                    continue
                
                index = int(choice)
                if index == 1:
                    ui.success(f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_assistant')}")
                    return ScenarioType.ASSISTANT
                elif index == 2:
                    ui.success(f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_group_chat')}")
                    return ScenarioType.GROUP_CHAT
                else:
                    ui.error(f"✗ {texts.get('invalid_input_number')}")
            
            except ValueError:
                ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                return None


class GroupSelector:
    """群组选择器"""
    
    @staticmethod
    async def list_available_groups() -> List[Dict[str, Any]]:
        """列出所有可用的群组
        
        Returns:
            群组列表
        """
        groups = await query_all_groups_from_mongodb()
        
        for idx, group in enumerate(groups, start=1):
            group["index"] = idx
            group_id = group["group_id"]
            group["name"] = "group_chat" if group_id == "AI产品群" else group_id
        
        return groups
    
    @staticmethod
    async def select_group(groups: List[Dict[str, Any]], texts: I18nTexts) -> Optional[str]:
        """交互式选择群组
        
        Args:
            groups: 群组列表
            texts: 国际化文本对象
            
        Returns:
            选中的 group_id 或 None（取消）
        """
        from .ui import ChatUI
        
        if not groups:
            ChatUI.print_error(texts.get("groups_not_found"), texts)
            print(f"{texts.get('groups_extract_hint')}\n")
            return None
        
        ChatUI.print_group_list(groups, texts)
        
        while True:
            try:
                choice = input(f"\n{texts.get('groups_select_prompt')} [1-{len(groups)}]: ").strip()
                if not choice:
                    continue
                
                index = int(choice)
                if 1 <= index <= len(groups):
                    return groups[index - 1]["group_id"]
                else:
                    ChatUI.print_error(
                        texts.get("groups_select_range_error", min=1, max=len(groups)),
                        texts,
                    )
            
            except ValueError:
                ChatUI.print_error(texts.get("invalid_input_number"), texts)
            except KeyboardInterrupt:
                print("\n")
                ChatUI.print_info(texts.get("groups_selection_cancelled"), texts)
                return None

