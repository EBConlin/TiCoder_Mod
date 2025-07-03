#!/usr/bin/env python3
"""
TiCoder query_models.py Integration with Haskell Generation
Integrates Haskell JSON AST generation and deterministic translation with full diagnostics.
"""

import json
import time
import hashlib
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import our deterministic translator
from haskell_python_translator import GuaranteedTranslationSystem

# Import existing TiCoder modules
import config
from config import debug_print
from static_mutation import prune_equivalent_codes


class DiagnosticLevel(Enum):
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class TranslationDiagnostic:
    """Diagnostic information for translation process"""
    timestamp: str
    level: DiagnosticLevel
    stage: str  # "haskell_generation", "translation", "validation", etc.
    message: str
    data: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None


@dataclass
class HaskellGenerationResult:
    """Result of Haskell generation attempt"""
    success: bool
    haskell_json: Optional[str]
    python_code: Optional[str]
    is_deterministic: bool
    method_used: str
    confidence: float
    diagnostics: List[TranslationDiagnostic]
    error_message: Optional[str] = None


class HaskellTiCoderIntegration:
    """Main integration class for TiCoder with comprehensive diagnostics"""
    
    def __init__(self):
        self.translation_system = GuaranteedTranslationSystem()
        self.diagnostics_log = []
        self.haskell_cache = {}  # Cache successful Haskell generations
        self.session_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'deterministic_translations': 0,
            'fallback_translations': 0,
            'cache_hits': 0,
            'start_time': datetime.now()
        }
    
    def log_diagnostic(self, level: DiagnosticLevel, stage: str, message: str, 
                      data: Optional[Dict] = None, duration: Optional[float] = None):
        """Log diagnostic information"""
        diagnostic = TranslationDiagnostic(
            timestamp=datetime.now().isoformat(),
            level=level,
            stage=stage,
            message=message,
            data=data,
            duration_ms=duration
        )
        self.diagnostics_log.append(diagnostic)
        
        # Also use TiCoder's debug_print for consistency
        if level in [DiagnosticLevel.WARNING, DiagnosticLevel.ERROR]:
            debug_print(f"[{level.value.upper()}] {stage}: {message}")
        elif config.verbosity > 0:
            debug_print(f"[{level.value.upper()}] {stage}: {message}")
    
    def enhanced_get_code_suggestions(self, client, prog_data: Dict, tests_in_ctxt: List[str] = None, 
                                    token_counter=None) -> Tuple[List[str], List[str]]:
        """
        Enhanced version of get_code_suggestions that generates Haskell first.
        
        Returns:
            Tuple of (original_haskell_codes, translated_python_codes)
        """
        start_time = time.time()
        self.session_stats['total_attempts'] += 1
        
        self.log_diagnostic(DiagnosticLevel.INFO, "code_generation", 
                          f"Starting Haskell code generation for function: {prog_data.get('func_name', 'unknown')}")
        
        try:
            # Generate Haskell JSON AST suggestions
            haskell_results = self._generate_haskell_suggestions(
                client, prog_data, tests_in_ctxt, token_counter
            )
            
            # Translate to Python and collect results
            original_codes = []
            translated_codes = []
            translation_summary = {
                'total_generated': len(haskell_results),
                'successful_translations': 0,
                'deterministic_translations': 0,
                'fallback_translations': 0,
                'failed_translations': 0
            }
            
            for result in haskell_results:
                if result.success:
                    original_codes.append(result.haskell_json)
                    translated_codes.append(result.python_code)
                    translation_summary['successful_translations'] += 1
                    
                    if result.is_deterministic:
                        translation_summary['deterministic_translations'] += 1
                        self.session_stats['deterministic_translations'] += 1
                    else:
                        translation_summary['fallback_translations'] += 1
                        self.session_stats['fallback_translations'] += 1
                else:
                    translation_summary['failed_translations'] += 1
                
                # Add result diagnostics to our log
                self.diagnostics_log.extend(result.diagnostics)
            
            # Prune equivalent Python codes (keeping Haskell mapping)
            if translated_codes:
                original_codes, translated_codes = self._prune_equivalent_with_mapping(
                    original_codes, translated_codes
                )
            
            # Log final summary
            duration = (time.time() - start_time) * 1000
            self.log_diagnostic(DiagnosticLevel.INFO, "code_generation", 
                              f"Completed code generation", 
                              data=translation_summary, duration=duration)
            
            self.session_stats['successful_generations'] += len(translated_codes)
            
            return original_codes, translated_codes
            
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            self.log_diagnostic(DiagnosticLevel.ERROR, "code_generation", error_msg,
                              data={'exception': str(e), 'traceback': traceback.format_exc()})
            
            # Return empty lists on failure
            return [], []
    
    def _generate_haskell_suggestions(self, client, prog_data: Dict, tests_in_ctxt: List[str] = None,
                                    token_counter=None, num_suggestions: int = None) -> List[HaskellGenerationResult]:
        """Generate Haskell JSON AST suggestions from LLM"""
        
        if num_suggestions is None:
            num_suggestions = config.MAX_NUM_CODEX_CODE_SUGGESTIONS
        
        # Build prompt for Haskell generation
        prompt = self._build_haskell_generation_prompt(prog_data, tests_in_ctxt)
        
        # Check cache first
        cache_key = self._compute_prompt_cache_key(prompt)
        if cache_key in self.haskell_cache:
            self.session_stats['cache_hits'] += 1
            self.log_diagnostic(DiagnosticLevel.INFO, "haskell_generation", 
                              "Using cached Haskell generation")
            return self.haskell_cache[cache_key]
        
        self.log_diagnostic(DiagnosticLevel.INFO, "haskell_generation", 
                          f"Generating {num_suggestions} Haskell suggestions")
        
        # Query LLM for Haskell
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=prompt,
                max_tokens=config.MAX_TOKENS,
                n=num_suggestions,
                temperature=config.sampling_temperature,
            )
            
            query_duration = (time.time() - start_time) * 1000
            if token_counter:
                token_counter.add_tokens(response.usage.total_tokens)
            
            self.log_diagnostic(DiagnosticLevel.INFO, "haskell_generation", 
                              f"LLM query completed", 
                              data={'tokens_used': response.usage.total_tokens}, 
                              duration=query_duration)
            
        except Exception as e:
            error_msg = f"LLM query failed: {str(e)}"
            self.log_diagnostic(DiagnosticLevel.ERROR, "haskell_generation", error_msg)
            return []
        
        # Process each response
        results = []
        for i, choice in enumerate(response.choices):
            result = self._process_haskell_response(choice.message.content, prog_data, i)
            results.append(result)
        
        # Cache successful results
        if results:
            self.haskell_cache[cache_key] = results
        
        return results
    
    def _process_haskell_response(self, response_content: str, prog_data: Dict, 
                                response_index: int) -> HaskellGenerationResult:
        """Process a single Haskell response from LLM"""
        
        diagnostics = []
        start_time = time.time()

        print("=== Raw LLM Response ===")
        print(response_content)
        print("========================")
        
        def add_diagnostic(level: DiagnosticLevel, message: str, data: Optional[Dict] = None):
            diagnostics.append(TranslationDiagnostic(
                timestamp=datetime.now().isoformat(),
                level=level,
                stage=f"process_response_{response_index}",
                message=message,
                data=data
            ))
        
        try:
            # Extract JSON from response
            haskell_json = self._extract_json_from_response(response_content)
            if not haskell_json:
                add_diagnostic(DiagnosticLevel.ERROR, "Could not extract valid JSON from LLM response",
                             data={'response_preview': response_content[:200]})
                return HaskellGenerationResult(
                    success=False, haskell_json=None, python_code=None,
                    is_deterministic=False, method_used="extraction_failed",
                    confidence=0.0, diagnostics=diagnostics, 
                    error_message="JSON extraction failed"
                )
            
            add_diagnostic(DiagnosticLevel.INFO, "Successfully extracted JSON from response")
            
            # Validate JSON structure
            validation_result = self._validate_haskell_json(haskell_json)
            if not validation_result['valid']:
                add_diagnostic(DiagnosticLevel.ERROR, "Haskell JSON validation failed",
                             data=validation_result)
                return HaskellGenerationResult(
                    success=False, haskell_json=haskell_json, python_code=None,
                    is_deterministic=False, method_used="validation_failed",
                    confidence=0.0, diagnostics=diagnostics,
                    error_message=validation_result['error']
                )
            
            add_diagnostic(DiagnosticLevel.INFO, "Haskell JSON validation passed",
                         data={'validation_details': validation_result})
            
            # Translate to Python
            translation_start = time.time()
            translation_result = self.translation_system.translate_with_guarantees(
                haskell_json, prog_data.get('ctxt', '')
            )
            translation_duration = (time.time() - translation_start) * 1000
            
            add_diagnostic(DiagnosticLevel.INFO, "Translation completed",
                         data={
                             'deterministic': translation_result['is_deterministic'],
                             'method': translation_result['method_used'],
                             'confidence': translation_result['confidence']
                         }, duration=translation_duration)
            
            # Validate Python code
            if not self._validate_python_code(translation_result['python_code']):
                add_diagnostic(DiagnosticLevel.ERROR, "Generated Python code is invalid")
                return HaskellGenerationResult(
                    success=False, haskell_json=haskell_json, 
                    python_code=translation_result['python_code'],
                    is_deterministic=translation_result['is_deterministic'],
                    method_used=translation_result['method_used'],
                    confidence=0.0, diagnostics=diagnostics,
                    error_message="Invalid Python code generated"
                )
            
            total_duration = (time.time() - start_time) * 1000
            add_diagnostic(DiagnosticLevel.INFO, "Processing completed successfully",
                         duration=total_duration)
            
            return HaskellGenerationResult(
                success=True,
                haskell_json=haskell_json,
                python_code=translation_result['python_code'],
                is_deterministic=translation_result['is_deterministic'],
                method_used=translation_result['method_used'],
                confidence=translation_result['confidence'],
                diagnostics=diagnostics
            )
            
        except Exception as e:
            add_diagnostic(DiagnosticLevel.ERROR, f"Processing failed with exception: {str(e)}",
                         data={'exception': str(e), 'traceback': traceback.format_exc()})
            return HaskellGenerationResult(
                success=False, haskell_json=None, python_code=None,
                is_deterministic=False, method_used="exception",
                confidence=0.0, diagnostics=diagnostics,
                error_message=str(e)
            )
    
    def _build_haskell_generation_prompt(self, prog_data: Dict, tests_in_ctxt: List[str] = None) -> List[Dict[str, str]]:
        """Build prompt for Haskell JSON AST generation"""
        
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a functional programming expert. Generate Safe Haskell code in JSON AST format. "
                    "CRITICAL: Use only Safe Haskell - no IO, no unsafe operations, only pure functions. "
                    "Focus on list comprehensions, map, filter, fold, basic recursion, and lambda expressions. "
                    "Always respond with valid JSON AST using the exact structure from examples. "
                    "Do NOT include explanations - only the JSON AST."
                )
            }
        ]
        
        # Build user prompt
        context = prog_data.get('ctxt', '')
        signature = prog_data.get('sig', '')
        func_name = prog_data.get('func_name', 'unknown_function')
        
        user_content = f"Generate Safe Haskell function '{func_name}' as JSON AST:\n\n"
        
        if context.strip():
            user_content += f"Context:\n{context}\n\n"
        
        user_content += f"Function specification:\n{signature}\n\n"
        
        if tests_in_ctxt and len(tests_in_ctxt) > 0:
            user_content += "Example tests for reference:\n"
            for test in tests_in_ctxt[:2]:  # Limit to avoid token overflow
                user_content += f"{test}\n"
            user_content += "\n"
        
        user_content += self._get_json_ast_examples()
        user_content += f"\n\nGenerate function '{func_name}' as JSON AST. Respond ONLY with JSON."
        
        prompt.append({"role": "user", "content": user_content})
        return prompt
    
    def _get_json_ast_examples(self) -> str:
        """Get JSON AST format examples"""
        return '''
JSON AST Examples:

List comprehension:
{
  "type": "FunctionDefinition",
  "name": "squares",
  "parameters": [{"name": "xs"}],
  "body": {
    "type": "ListComprehension", 
    "expression": {
      "type": "BinaryOp",
      "operator": "*",
      "left": {"type": "Variable", "name": "x"},
      "right": {"type": "Variable", "name": "x"}
    },
    "generators": [
      {"type": "Generator", "variable": "x", "source": {"type": "Variable", "name": "xs"}}
    ]
  }
}

Map function:
{
  "type": "FunctionDefinition",
  "name": "increment_all",
  "parameters": [{"name": "xs"}],
  "body": {
    "type": "Application",
    "function": {"type": "Variable", "name": "map"},
    "arguments": [
      {
        "type": "Lambda",
        "parameters": [{"name": "x"}],
        "body": {
          "type": "BinaryOp",
          "operator": "+", 
          "left": {"type": "Variable", "name": "x"},
          "right": {"type": "Literal", "value": 1, "literalType": "Integer"}
        }
      },
      {"type": "Variable", "name": "xs"}
    ]
  }
}'''
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON from LLM response with comprehensive parsing"""
        response = response.strip()
        
        # Try direct JSON parse first
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        import re
        
        # Look for ```json blocks
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                json_content = match.group(1).strip()
                json.loads(json_content)
                return json_content
            except json.JSONDecodeError:
                pass
        
        # Look for generic ``` blocks
        code_pattern = r'```\s*\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        if match:
            try:
                json_content = match.group(1).strip()
                json.loads(json_content)
                return json_content
            except json.JSONDecodeError:
                pass
        
        # Find JSON object boundaries
        start = response.find('{')
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            json_content = response[start:i+1]
                            json.loads(json_content)
                            return json_content
                        except json.JSONDecodeError:
                            break
        
        return None
    
    def _validate_haskell_json(self, haskell_json: str) -> Dict[str, Any]:
        """Validate Haskell JSON AST structure"""
        try:
            ast_data = json.loads(haskell_json)
        except json.JSONDecodeError as e:
            return {'valid': False, 'error': f'Invalid JSON: {str(e)}'}
        
        # Check required top-level structure
        if not isinstance(ast_data, dict):
            return {'valid': False, 'error': 'AST must be a JSON object'}
        
        if ast_data.get('type') != 'FunctionDefinition':
            return {'valid': False, 'error': 'Top level must be FunctionDefinition'}
        
        if 'name' not in ast_data:
            return {'valid': False, 'error': 'Function name is required'}
        
        if 'body' not in ast_data:
            return {'valid': False, 'error': 'Function body is required'}
        
        # Validate that it uses Safe Haskell constructs
        unsafe_constructs = self._check_for_unsafe_constructs(ast_data)
        if unsafe_constructs:
            return {'valid': False, 'error': f'Unsafe constructs found: {unsafe_constructs}'}
        
        return {
            'valid': True, 
            'function_name': ast_data['name'],
            'parameter_count': len(ast_data.get('parameters', [])),
            'body_type': ast_data['body'].get('type', 'unknown')
        }
    
    def _check_for_unsafe_constructs(self, ast_data: Dict) -> List[str]:
        """Check for unsafe Haskell constructs"""
        unsafe_constructs = []
        
        def check_node(node):
            if isinstance(node, dict):
                node_type = node.get('type', '')
                
                # Check for IO operations
                if node_type in ['IOAction', 'MonadicBind']:
                    unsafe_constructs.append('IO operations')
                
                # Check for unsafe functions
                if node_type == 'Variable' and node.get('name', '').startswith('unsafe'):
                    unsafe_constructs.append(f"Unsafe function: {node.get('name')}")
                
                # Check for FFI
                if node_type in ['ForeignImport', 'ForeignExport']:
                    unsafe_constructs.append('Foreign Function Interface')
                
                # Recursively check children
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        check_node(value)
            
            elif isinstance(node, list):
                for item in node:
                    check_node(item)
        
        check_node(ast_data)
        return unsafe_constructs
    
    def _validate_python_code(self, python_code: str) -> bool:
        """Validate that generated Python code is syntactically correct"""
        try:
            import ast
            ast.parse(python_code)
            return True
        except SyntaxError:
            return False
    
    def _compute_prompt_cache_key(self, prompt: List[Dict[str, str]]) -> str:
        """Compute cache key for prompt"""
        prompt_str = json.dumps(prompt, sort_keys=True)
        return hashlib.md5(prompt_str.encode()).hexdigest()
    
    def _prune_equivalent_with_mapping(self, haskell_codes: List[str], 
                                     python_codes: List[str]) -> Tuple[List[str], List[str]]:
        """Prune equivalent Python codes while maintaining Haskell mapping"""
        if len(haskell_codes) != len(python_codes):
            self.log_diagnostic(DiagnosticLevel.WARNING, "pruning", 
                              "Mismatched Haskell/Python code counts")
            return haskell_codes, python_codes
        
        # Use TiCoder's existing pruning on Python codes
        unique_python = []
        corresponding_haskell = []
        
        for h_code, p_code in zip(haskell_codes, python_codes):
            if p_code not in unique_python:
                unique_python.append(p_code)
                corresponding_haskell.append(h_code)
        
        pruned_count = len(python_codes) - len(unique_python)
        if pruned_count > 0:
            self.log_diagnostic(DiagnosticLevel.INFO, "pruning", 
                              f"Pruned {pruned_count} equivalent Python codes")
        
        return corresponding_haskell, unique_python
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        return {
            'session_stats': self.session_stats,
            'translation_system_stats': self.translation_system.get_system_stats(),
            'recent_diagnostics': [asdict(d) for d in self.diagnostics_log[-20:]],
            'cache_stats': {
                'haskell_cache_size': len(self.haskell_cache),
                'cache_hit_rate': self.session_stats['cache_hits'] / max(self.session_stats['total_attempts'], 1)
            },
            'error_summary': self._get_error_summary(),
            'performance_metrics': self._get_performance_metrics()
        }
    
    def _get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type"""
        error_counts = {}
        for diagnostic in self.diagnostics_log:
            if diagnostic.level == DiagnosticLevel.ERROR:
                stage = diagnostic.stage
                error_counts[stage] = error_counts.get(stage, 0) + 1
        return error_counts
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        durations = [d.duration_ms for d in self.diagnostics_log if d.duration_ms is not None]
        
        if not durations:
            return {'no_timing_data': True}
        
        return {
            'avg_processing_time_ms': sum(durations) / len(durations),
            'max_processing_time_ms': max(durations),
            'min_processing_time_ms': min(durations),
            'total_operations': len(durations)
        }
    
    def save_diagnostics_report(self, filepath: str):
        """Save detailed diagnostics report to file"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'comprehensive_diagnostics': self.get_comprehensive_diagnostics(),
            'full_diagnostic_log': [asdict(d) for d in self.diagnostics_log]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log_diagnostic(DiagnosticLevel.INFO, "diagnostics", 
                          f"Saved diagnostics report to {filepath}")


# Integration function to replace existing get_code_suggestions
def get_enhanced_code_suggestions(client, prog_data, tests_in_ctxt=None, token_counter=None):
    """
    Drop-in replacement for TiCoder's get_code_suggestions with Haskell generation.
    
    This function maintains the same signature and behavior as the original
    but adds Haskell generation and deterministic translation.
    """
    # Create global instance (or reuse existing one)
    if not hasattr(get_enhanced_code_suggestions, '_integration_instance'):
        get_enhanced_code_suggestions._integration_instance = HaskellTiCoderIntegration()
    
    integration = get_enhanced_code_suggestions._integration_instance
    
    # Use enhanced generation
    haskell_codes, python_codes = integration.enhanced_get_code_suggestions(
        client, prog_data, tests_in_ctxt, token_counter
    )
    
    # Store Haskell codes for later retrieval (in case we need them)
    if not hasattr(config, 'current_haskell_codes'):
        config.current_haskell_codes = {}
    
    config.current_haskell_codes[prog_data.get('func_name', 'unknown')] = haskell_codes
    
    # Return Python codes to maintain TiCoder compatibility
    return python_codes, python_codes  # Return as (orig_codes, codes) tuple


# Diagnostic access function
def get_current_diagnostics():
    """Get current diagnostic information"""
    if hasattr(get_enhanced_code_suggestions, '_integration_instance'):
        return get_enhanced_code_suggestions._integration_instance.get_comprehensive_diagnostics()
    return {'no_integration_instance': True}


# Function to save diagnostics (can be called from main.py)
def save_diagnostics_if_available(filepath: str):
    """Save diagnostics if integration is active"""
    if hasattr(get_enhanced_code_suggestions, '_integration_instance'):
        get_enhanced_code_suggestions._integration_instance.save_diagnostics_report(filepath)
        return True
    return False


# Test function
def test_integration():
    """Test the TiCoder integration"""
    
    # Mock client for testing
    class MockClient:
        class ChatCompletions:
            def create(self, **kwargs):
                class MockResponse:
                    def __init__(self):
                        self.choices = [self._create_choice()]
                        self.usage = type('Usage', (), {'total_tokens': 150})()
                    
                    def _create_choice(self):
                        mock_haskell = {
                            "type": "FunctionDefinition",
                            "name": "test_function",
                            "parameters": [{"name": "xs"}],
                            "body": {
                                "type": "Application",
                                "function": {"type": "Variable", "name": "map"},
                                "arguments": [
                                    {
                                        "type": "Lambda",
                                        "parameters": [{"name": "x"}],
                                        "body": {
                                            "type": "BinaryOp",
                                            "operator": "*",
                                            "left": {"type": "Variable", "name": "x"},
                                            "right": {"type": "Literal", "value": 2, "literalType": "Integer"}
                                        }
                                    },
                                    {"type": "Variable", "name": "xs"}
                                ]
                            }
                        }
                        
                        return type('Choice', (), {
                            'message': type('Message', (), {
                                'content': json.dumps(mock_haskell)
                            })()
                        })()
                
                return MockResponse()
        
        def __init__(self):
            self.chat = type('Chat', (), {'completions': self.ChatCompletions()})()
    
    # Test data
    prog_data = {
        'func_name': 'double_list',
        'sig': 'def double_list(xs):\n    """Double all elements in the list"""',
        'ctxt': ''
    }
    
    # Run test
    mock_client = MockClient()
    orig_codes, codes = get_enhanced_code_suggestions(mock_client, prog_data)
    
    print("=== TiCoder Integration Test ===")
    print(f"Generated {len(codes)} Python code suggestions")
    for i, code in enumerate(codes):
        print(f"\nCode {i+1}:")
        print(code)
    
    # Print diagnostics
    diagnostics = get_current_diagnostics()
    print(f"\n=== Diagnostics Summary ===")
    print(f"Session stats: {diagnostics.get('session_stats', {})}")
    print(f"Error summary: {diagnostics.get('error_summary', {})}")


if __name__ == "__main__":
    test_integration()
