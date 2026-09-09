import os
"""Main inference pipeline with tool support"""
from typing import Dict, List, Optional
from src.transformer.model import MathTransformerModel, DEFAULT_LORA_ADAPTER_PATH
from src.generation.generator import MathGenerator
from src.generation.prompts import PromptTemplate
from src.input_processing import UniversalMathInputProcessor
from src.output.formatter import OutputFormatter
from src.tools.tool_registry import tool_registry

# --- NEW: Import the specific tools ---
from src.tools.sympy_solver import SymPySolver
from src.tools.numpy_calculator import NumpyCalculator
from src.tools.matplotlib_plotter import MatplotlibPlotter
from src.tools.code_executor import CodeExecutor
from src.tools.wolfram_alpha import WolframAlphaTool

class MathSolverInference:
    """Complete inference pipeline with tool calling"""
    
    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
        # BUG FIX: was the Windows-only literal ".\models\lora_adapter", which
        # silently failed to resolve on Linux/Colab (see model.py for details).
        # Now reuses the single source of truth defined in model.py so the two
        # defaults can't drift out of sync again.
        lora_adapter_path: str = DEFAULT_LORA_ADAPTER_PATH,
        enable_tools: bool = True,
        enable_wolfram: bool = False,
        wolfram_api_key: str = None
    ):
        print("🚀 Initializing Math Solver Pipeline...")
        if wolfram_api_key is None:
            wolfram_api_key = os.getenv('WOLFRAM_API_KEY')
        # Initialize components
        self.input_processor = UniversalMathInputProcessor()
        self.model_wrapper = MathTransformerModel(
            base_model_id=base_model_id,
            lora_adapter_path=lora_adapter_path
        )
        self.generator = MathGenerator(self.model_wrapper)
        self.output_formatter = OutputFormatter()
        self.enable_tools = enable_tools
        self.enable_wolfram = enable_wolfram
        
        # --- NEW: Register the tools if enabled ---
        if enable_tools:
            print("⚙️  Registering tools...")
            # We must manually register each tool we want the Agent to use
            tool_registry.register(SymPySolver())
            tool_registry.register(NumpyCalculator())
            tool_registry.register(MatplotlibPlotter())
            tool_registry.register(CodeExecutor())

            if enable_wolfram:
                if wolfram_api_key:
                    tool_registry.register(WolframAlphaTool(app_id=wolfram_api_key))
                    print("✅ Wolfram Alpha enabled")
                else:
                    print("⚠️  Wolfram Alpha enabled but no API key provided")
                    print("   Set wolfram_api_key parameter or get key at:")
                    print("   https://products.wolframalpha.com/api/")
            else:
                print("ℹ️  Wolfram Alpha disabled")
            
            print(f"🔧 Tools enabled: {list(tool_registry.list_tools().keys())}")
        
        print("✅ Math Solver initialized and ready!")
    
    def solve(
        self,
        problem: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_tools: bool = None,
        return_raw: bool = False
    ) -> Dict:
        """Solve a math problem with optional tool calling"""
        
        use_tools = use_tools if use_tools is not None else self.enable_tools
        if system_prompt is None:
            system_prompt = "with_tools" if use_tools else "step_by_step"
        
        # 1. Process input
        processed_problem = self.input_processor.process(problem)
        
        # 2. Create messages
        messages = PromptTemplate.create_messages(
            processed_problem,
            system_prompt=system_prompt
        )
        
        # 3. Generate solution (with or without tools)
        if use_tools:
            generation_result = self.generator.generate_with_tools(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            answer = generation_result["final_answer"]
            tool_calls = generation_result.get("tool_calls", [])
        else:
            raw_output = self.generator.generate(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            answer = self.generator.extract_answer(raw_output)
            tool_calls = []
        
        # 4. Format output
        formatted_output = self.output_formatter.format(answer)
        final_answer = self.output_formatter.extract_final_answer(answer)
        
        result = {
            "problem": processed_problem,
            "solution": answer,
            "formatted": formatted_output,
            "final_answer": final_answer,
            "tool_calls": tool_calls,
            "tools_used": len(tool_calls) > 0
        }
        
        if return_raw and not use_tools:
            result["raw_output"] = raw_output
        
        return result
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools"""
        return tool_registry.list_tools()