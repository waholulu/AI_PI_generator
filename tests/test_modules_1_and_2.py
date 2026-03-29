import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 添加项目根目录到 sys.path, 以免找不到 agents 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestrator import ResearchState
from agents.ideation_agent import ideation_node
from agents.literature_agent import literature_node

def test_module_1_and_2():
    print("=== 开始测试 Module 1: Ideation ===")
    
    # 确保输出目录存在
    os.makedirs("config", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("data/literature", exist_ok=True)

    state = ResearchState(
        domain_input="GeoAI and Urban Planning",
        execution_status="starting"
    )
    
    # 模拟执行模块 1
    state = ideation_node(state)
    print(f"Module 1 结束状态: {state.get('execution_status')}")
    
    # 检查模块 1 输出
    if not os.path.exists("config/research_plan.json"):
        print("❌ 错误: Research plan (config/research_plan.json) 未生成！")
        return False
    
    print("\n=== 开始测试 Module 2: Literature Harvester ===")
    # 直接将模块 1 修改后的 state 传给模块 2
    state = literature_node(state)
    print(f"Module 2 结束状态: {state.get('execution_status')}")
    
    # 检查模块 2 输出
    missing_files = []
    if not os.path.exists("data/literature/index.json"):
        missing_files.append("data/literature/index.json")
    if not os.path.exists("output/references.bib"):
        missing_files.append("output/references.bib")
        
    if missing_files:
        print(f"❌ 错误: 缺少生成文件: {', '.join(missing_files)}")
        return False
        
    print("\n✅ Module 1 和 2 连续测试通过！相关文件已生成。")
    return True

if __name__ == "__main__":
    success = test_module_1_and_2()
    if not success:
        sys.exit(1)
