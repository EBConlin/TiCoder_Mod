#!/usr/bin/env python3
"""
Deterministic Rule-Based Haskell JSON AST to Python Translator
Uses deterministic rules for common patterns, LLM fallback for complex cases.
"""

import ast
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class TranslationResult(Enum):
    """Result status of translation attempt"""
    SUCCESS = "success"
    FALLBACK_NEEDED = "fallback_needed"
    FAILED = "failed"


@dataclass
class TranslationAttempt:
    """Result of a translation attempt"""
    status: TranslationResult
    result: Optional[ast.AST]
    error_message: Optional[str] = None
    rule_used: Optional[str] = None


class DeterministicHaskellTranslator:
    """Rule-based translator with deterministic guarantees for core patterns"""
    
    def __init__(self):
        # Track which rules succeeded/failed for learning
        self.rule_success_stats = {}
        self.fallback_cache = {}
        
        # Core deterministic translation rules
        self.rules = {
            'list_comprehension': self._rule_list_comprehension,
            'map_application': self._rule_map_application, 
            'filter_application': self._rule_filter_application,
            'fold_application': self._rule_fold_application,
            'simple_function': self._rule_simple_function,
            'lambda_expression': self._rule_lambda_expression,
            'basic_arithmetic': self._rule_basic_arithmetic,
            'conditional_expression': self._rule_conditional_expression,
            'variable_reference': self._rule_variable_reference,
            'literal_value': self._rule_literal_value,
            'function_composition': self._rule_function_composition,
        }
        
        # Deterministic built-in mappings (guaranteed to work)
        self.guaranteed_builtins = {
            'length': ('len', lambda args: f"len({args[0]})"),
            'head': ('_head', lambda args: f"({args[0]}[0] if {args[0]} else None)"),
            'tail': ('_tail', lambda args: f"{args[0]}[1:]"),
            'reverse': ('list(reversed', lambda args: f"list(reversed({args[0]}))"),
            'sum': ('sum', lambda args: f"sum({args[0]})"),
            'null': ('_is_empty', lambda args: f"len({args[0]}) == 0"),
            'take': ('_take', lambda args: f"{args[1]}[:{args[0]}]"),
            'drop': ('_drop', lambda args: f"{args[1]}[{args[0]}:]"),
        }
    
    def translate_with_guarantees(self, haskell_json: Union[str, Dict]) -> Tuple[str, bool, str]:
        """
        Main translation with deterministic guarantees.
        
        Returns:
            Tuple of (python_code, is_deterministic, method_used)
            - python_code: Generated Python code
            - is_deterministic: True if rule-based, False if LLM fallback
            - method_used: Name of rule or "llm_fallback"
        """
        if isinstance(haskell_json, str):
            haskell_ast = json.loads(haskell_json)
        else:
            haskell_ast = haskell_json
        
        # Try deterministic rules first
        attempt = self._try_deterministic_translation(haskell_ast)
        
        if attempt.status == TranslationResult.SUCCESS:
            python_code = ast.unparse(attempt.result)
            self._record_rule_success(attempt.rule_used)
            return python_code, True, attempt.rule_used
        
        # Fall back to LLM for complex cases
        elif attempt.status == TranslationResult.FALLBACK_NEEDED:
            python_code = self._llm_fallback_translation(haskell_ast, attempt.error_message)
            return python_code, False, "llm_fallback"
        
        else:
            # Complete failure - return safe placeholder
            return self._generate_safe_placeholder(), False, "failed"
    
    def _try_deterministic_translation(self, node: Dict[str, Any]) -> TranslationAttempt:
        """Try to translate using deterministic rules"""
        node_type = node.get('type', '').lower()
        
        # Try each rule in priority order
        rule_priority = [
            ('literal_value', self._is_literal),
            ('variable_reference', self._is_variable),
            ('list_comprehension', self._is_list_comprehension),
            ('map_application', self._is_map_application),
            ('filter_application', self._is_filter_application),
            ('fold_application', self._is_fold_application),
            ('lambda_expression', self._is_lambda),
            ('basic_arithmetic', self._is_basic_arithmetic),
            ('conditional_expression', self._is_conditional),
            ('simple_function', self._is_simple_function),
            ('function_composition', self._is_function_composition),
        ]
        
        for rule_name, predicate in rule_priority:
            if predicate(node):
                try:
                    result_ast = self.rules[rule_name](node)
                    if result_ast:
                        return TranslationAttempt(
                            status=TranslationResult.SUCCESS,
                            result=result_ast,
                            rule_used=rule_name
                        )
                except Exception as e:
                    # Rule failed, continue to next rule
                    continue
        
        # No rule matched - need fallback
        return TranslationAttempt(
            status=TranslationResult.FALLBACK_NEEDED,
            result=None,
            error_message=f"No deterministic rule for node type: {node_type}"
        )
    
    # Deterministic rule predicates
    def _is_literal(self, node: Dict) -> bool:
        return node.get('type') == 'Literal'
    
    def _is_variable(self, node: Dict) -> bool:
        return node.get('type') == 'Variable'
    
    def _is_list_comprehension(self, node: Dict) -> bool:
        return node.get('type') == 'ListComprehension'
    
    def _is_map_application(self, node: Dict) -> bool:
        return (node.get('type') == 'Application' and 
                node.get('function', {}).get('name') == 'map')
    
    def _is_filter_application(self, node: Dict) -> bool:
        return (node.get('type') == 'Application' and 
                node.get('function', {}).get('name') == 'filter')
    
    def _is_fold_application(self, node: Dict) -> bool:
        func_name = node.get('function', {}).get('name', '')
        return (node.get('type') == 'Application' and 
                func_name in ['foldl', 'foldr', 'fold'])
    
    def _is_lambda(self, node: Dict) -> bool:
        return node.get('type') == 'Lambda'
    
    def _is_basic_arithmetic(self, node: Dict) -> bool:
        return (node.get('type') == 'BinaryOp' and 
                node.get('operator') in ['+', '-', '*', '/', '//', '%', '**'])
    
    def _is_conditional(self, node: Dict) -> bool:
        return node.get('type') == 'Conditional'
    
    def _is_simple_function(self, node: Dict) -> bool:
        return node.get('type') == 'FunctionDefinition'
    
    def _is_function_composition(self, node: Dict) -> bool:
        return (node.get('type') == 'BinaryOp' and 
                node.get('operator') == '.')
    
    # Deterministic translation rules
    def _rule_literal_value(self, node: Dict) -> ast.AST:
        """Deterministic literal translation"""
        value = node['value']
        literal_type = node.get('literalType', 'unknown')
        
        if literal_type == 'Integer':
            return ast.Constant(value=int(value))
        elif literal_type == 'String':
            return ast.Constant(value=str(value))
        elif literal_type == 'Boolean':
            return ast.Constant(value=bool(value))
        elif literal_type == 'List':
            elements = [self._try_deterministic_translation(elem).result 
                       for elem in value]
            if all(elem is not None for elem in elements):
                return ast.List(elts=elements, ctx=ast.Load())
        
        raise ValueError(f"Cannot deterministically translate literal: {literal_type}")
    
    def _rule_variable_reference(self, node: Dict) -> ast.AST:
        """Deterministic variable reference"""
        return ast.Name(id=node['name'], ctx=ast.Load())
    
    def _rule_list_comprehension(self, node: Dict) -> ast.AST:
        """Deterministic list comprehension translation"""
        expr_attempt = self._try_deterministic_translation(node['expression'])
        if expr_attempt.status != TranslationResult.SUCCESS:
            raise ValueError("Cannot deterministically translate comprehension expression")
        
        generators = []
        for gen in node['generators']:
            if gen['type'] == 'Generator':
                target = ast.Name(id=gen['variable'], ctx=ast.Store())
                source_attempt = self._try_deterministic_translation(gen['source'])
                if source_attempt.status != TranslationResult.SUCCESS:
                    raise ValueError("Cannot deterministically translate generator source")
                
                generators.append(ast.comprehension(
                    target=target, 
                    iter=source_attempt.result, 
                    ifs=[], 
                    is_async=0
                ))
            elif gen['type'] == 'Guard':
                if generators:
                    guard_attempt = self._try_deterministic_translation(gen['condition'])
                    if guard_attempt.status != TranslationResult.SUCCESS:
                        raise ValueError("Cannot deterministically translate guard condition")
                    generators[-1].ifs.append(guard_attempt.result)
        
        return ast.ListComp(elt=expr_attempt.result, generators=generators)
    
    def _rule_map_application(self, node: Dict) -> ast.AST:
        """Deterministic map translation: map f xs -> [f(x) for x in xs]"""
        args = node.get('arguments', [])
        if len(args) != 2:
            raise ValueError("Map requires exactly 2 arguments")
        
        f_attempt = self._try_deterministic_translation(args[0])
        xs_attempt = self._try_deterministic_translation(args[1])
        
        if (f_attempt.status != TranslationResult.SUCCESS or 
            xs_attempt.status != TranslationResult.SUCCESS):
            raise ValueError("Cannot deterministically translate map arguments")
        
        # Create [f(x) for x in xs]
        return ast.ListComp(
            elt=ast.Call(func=f_attempt.result, args=[ast.Name(id='x', ctx=ast.Load())], keywords=[]),
            generators=[ast.comprehension(
                target=ast.Name(id='x', ctx=ast.Store()),
                iter=xs_attempt.result,
                ifs=[],
                is_async=0
            )]
        )
    
    def _rule_filter_application(self, node: Dict) -> ast.AST:
        """Deterministic filter translation: filter p xs -> [x for x in xs if p(x)]"""
        args = node.get('arguments', [])
        if len(args) != 2:
            raise ValueError("Filter requires exactly 2 arguments")
        
        p_attempt = self._try_deterministic_translation(args[0])
        xs_attempt = self._try_deterministic_translation(args[1])
        
        if (p_attempt.status != TranslationResult.SUCCESS or 
            xs_attempt.status != TranslationResult.SUCCESS):
            raise ValueError("Cannot deterministically translate filter arguments")
        
        # Create [x for x in xs if p(x)]
        return ast.ListComp(
            elt=ast.Name(id='x', ctx=ast.Load()),
            generators=[ast.comprehension(
                target=ast.Name(id='x', ctx=ast.Store()),
                iter=xs_attempt.result,
                ifs=[ast.Call(func=p_attempt.result, args=[ast.Name(id='x', ctx=ast.Load())], keywords=[])],
                is_async=0
            )]
        )
    
    def _rule_fold_application(self, node: Dict) -> ast.AST:
        """Deterministic fold translation using reduce"""
        func_name = node.get('function', {}).get('name', '')
        args = node.get('arguments', [])
        
        if len(args) < 2:
            raise ValueError("Fold requires at least 2 arguments")
        
        f_attempt = self._try_deterministic_translation(args[0])
        if f_attempt.status != TranslationResult.SUCCESS:
            raise ValueError("Cannot deterministically translate fold function")
        
        if len(args) == 2:
            # fold f xs
            xs_attempt = self._try_deterministic_translation(args[1])
            if xs_attempt.status != TranslationResult.SUCCESS:
                raise ValueError("Cannot deterministically translate fold list")
            
            return ast.Call(
                func=ast.Name(id='reduce', ctx=ast.Load()),
                args=[f_attempt.result, xs_attempt.result],
                keywords=[]
            )
        else:
            # fold f acc xs
            acc_attempt = self._try_deterministic_translation(args[1])
            xs_attempt = self._try_deterministic_translation(args[2])
            
            if (acc_attempt.status != TranslationResult.SUCCESS or 
                xs_attempt.status != TranslationResult.SUCCESS):
                raise ValueError("Cannot deterministically translate fold arguments")
            
            return ast.Call(
                func=ast.Name(id='reduce', ctx=ast.Load()),
                args=[f_attempt.result, xs_attempt.result, acc_attempt.result],
                keywords=[]
            )
    
    def _rule_lambda_expression(self, node: Dict) -> ast.AST:
        """Deterministic lambda translation"""
        params = node.get('parameters', [])
        body_attempt = self._try_deterministic_translation(node['body'])
        
        if body_attempt.status != TranslationResult.SUCCESS:
            raise ValueError("Cannot deterministically translate lambda body")
        
        args = [ast.arg(arg=param['name'], annotation=None) for param in params]
        
        return ast.Lambda(
            args=ast.arguments(
                args=args,
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=body_attempt.result
        )
    
    def _rule_basic_arithmetic(self, node: Dict) -> ast.AST:
        """Deterministic arithmetic operations"""
        left_attempt = self._try_deterministic_translation(node['left'])
        right_attempt = self._try_deterministic_translation(node['right'])
        
        if (left_attempt.status != TranslationResult.SUCCESS or 
            right_attempt.status != TranslationResult.SUCCESS):
            raise ValueError("Cannot deterministically translate arithmetic operands")
        
        op_mapping = {
            '+': ast.Add(),
            '-': ast.Sub(),
            '*': ast.Mult(),
            '/': ast.Div(),
            '//': ast.FloorDiv(),
            '%': ast.Mod(),
            '**': ast.Pow(),
        }
        
        operator = node['operator']
        if operator not in op_mapping:
            raise ValueError(f"Unknown arithmetic operator: {operator}")
        
        return ast.BinOp(
            left=left_attempt.result,
            op=op_mapping[operator],
            right=right_attempt.result
        )
    
    def _rule_conditional_expression(self, node: Dict) -> ast.AST:
        """Deterministic if-then-else translation"""
        test_attempt = self._try_deterministic_translation(node['condition'])
        then_attempt = self._try_deterministic_translation(node['thenBranch'])
        else_attempt = self._try_deterministic_translation(node['elseBranch'])
        
        if (test_attempt.status != TranslationResult.SUCCESS or 
            then_attempt.status != TranslationResult.SUCCESS or
            else_attempt.status != TranslationResult.SUCCESS):
            raise ValueError("Cannot deterministically translate conditional")
        
        return ast.IfExp(
            test=test_attempt.result,
            body=then_attempt.result,
            orelse=else_attempt.result
        )
    
    def _rule_simple_function(self, node: Dict) -> ast.AST:
        """Deterministic function definition translation"""
        name = node['name']
        params = node.get('parameters', [])
        body_attempt = self._try_deterministic_translation(node['body'])
        
        if body_attempt.status != TranslationResult.SUCCESS:
            raise ValueError("Cannot deterministically translate function body")
        
        args = [ast.arg(arg=param['name'], annotation=None) for param in params]
        
        # Wrap expression in return statement
        if isinstance(body_attempt.result, ast.expr):
            body_statements = [ast.Return(value=body_attempt.result)]
        else:
            body_statements = [body_attempt.result]
        
        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                args=args,
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=body_statements,
            decorator_list=[]
        )
    
    def _rule_function_composition(self, node: Dict) -> ast.AST:
        """Deterministic function composition: f . g -> lambda x: f(g(x))"""
        f_attempt = self._try_deterministic_translation(node['left'])
        g_attempt = self._try_deterministic_translation(node['right'])
        
        if (f_attempt.status != TranslationResult.SUCCESS or 
            g_attempt.status != TranslationResult.SUCCESS):
            raise ValueError("Cannot deterministically translate composition functions")
        
        # Create lambda x: f(g(x))
        x_var = ast.Name(id='x', ctx=ast.Load())
        g_call = ast.Call(func=g_attempt.result, args=[x_var], keywords=[])
        f_call = ast.Call(func=f_attempt.result, args=[g_call], keywords=[])
        
        return ast.Lambda(
            args=ast.arguments(
                args=[ast.arg(arg='x', annotation=None)],
                posonlyargs=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=f_call
        )
    
    def _llm_fallback_translation(self, haskell_ast: Dict, error_message: str) -> str:
        """LLM fallback for complex cases that rules can't handle"""
        # This would use an LLM to translate the complex case
        # For now, return a placeholder that indicates fallback was used
        
        # Cache key for this AST structure
        ast_str = json.dumps(haskell_ast, sort_keys=True)
        cache_key = hashlib.md5(ast_str.encode()).hexdigest()
        
        if cache_key in self.fallback_cache:
            return self.fallback_cache[cache_key]
        
        # In real implementation, this would call an LLM
        # For now, generate a safe placeholder
        placeholder = self._generate_fallback_placeholder(haskell_ast)
        self.fallback_cache[cache_key] = placeholder
        
        return placeholder
    
    def _generate_fallback_placeholder(self, haskell_ast: Dict) -> str:
        """Generate a safe placeholder for fallback cases"""
        func_name = haskell_ast.get('name', 'unknown_function')
        return f"""def {func_name}(*args):
    # Complex Haskell pattern - requires LLM translation
    # Original AST: {json.dumps(haskell_ast)[:100]}...
    raise NotImplementedError("Complex pattern needs LLM translation")"""
    
    def _generate_safe_placeholder(self) -> str:
        """Generate safe placeholder for complete failures"""
        return """def placeholder_function(*args):
    raise NotImplementedError("Translation failed")"""
    
    def _record_rule_success(self, rule_name: str):
        """Track which rules are working well"""
        if rule_name not in self.rule_success_stats:
            self.rule_success_stats[rule_name] = {'success': 0, 'total': 0}
        self.rule_success_stats[rule_name]['success'] += 1
        self.rule_success_stats[rule_name]['total'] += 1
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about translation success rates"""
        return {
            'rule_success_rates': {
                rule: (stats['success'] / stats['total']) 
                for rule, stats in self.rule_success_stats.items()
            },
            'fallback_cache_size': len(self.fallback_cache),
            'total_translations': sum(stats['total'] for stats in self.rule_success_stats.values())
        }


# Integration class for TiCoder
class GuaranteedTranslationSystem:
    """Integration layer that provides translation guarantees"""
    
    def __init__(self):
        self.deterministic_translator = DeterministicHaskellTranslator()
        self.translation_log = []
    
    def translate_with_guarantees(self, haskell_json: str, context: str = "") -> Dict[str, Any]:
        """
        Translate with detailed guarantees about the process.
        
        Returns:
            Dict containing:
            - python_code: The translated code
            - is_deterministic: Whether translation used deterministic rules
            - method_used: Which rule/method was used
            - confidence: Confidence level (1.0 for deterministic, <1.0 for LLM)
            - required_imports: List of imports needed
        """
        python_code, is_deterministic, method_used = self.deterministic_translator.translate_with_guarantees(haskell_json)
        
        # Add context and imports
        if context.strip():
            python_code = f"{context.strip()}\n\n{python_code}"
        
        required_imports = self._extract_required_imports(python_code)
        if required_imports:
            python_code = f"{required_imports}\n\n{python_code}"
        
        # Calculate confidence
        confidence = 1.0 if is_deterministic else 0.7  # Lower confidence for LLM fallback
        
        result = {
            'python_code': python_code,
            'is_deterministic': is_deterministic,
            'method_used': method_used,
            'confidence': confidence,
            'required_imports': required_imports,
        }
        
        # Log for analysis
        self.translation_log.append({
            'haskell_hash': hashlib.md5(haskell_json.encode()).hexdigest()[:8],
            'method': method_used,
            'deterministic': is_deterministic,
            'confidence': confidence,
        })
        
        return result
    
    def _extract_required_imports(self, python_code: str) -> str:
        """Extract required imports based on code analysis"""
        imports = []
        
        if 'reduce(' in python_code:
            imports.append('from functools import reduce')
        if any(func in python_code for func in ['itertools.', 'permutations', 'combinations']):
            imports.append('import itertools')
        if 'math.' in python_code:
            imports.append('import math')
        
        return '\n'.join(imports)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        translator_stats = self.deterministic_translator.get_translation_stats()
        
        recent_translations = self.translation_log[-100:]  # Last 100
        deterministic_rate = sum(1 for t in recent_translations if t['deterministic']) / max(len(recent_translations), 1)
        
        return {
            'deterministic_success_rate': deterministic_rate,
            'total_translations': len(self.translation_log),
            'rule_performance': translator_stats['rule_success_rates'],
            'fallback_usage': 1 - deterministic_rate,
        }


# Test the deterministic system
def test_deterministic_translation():
    """Test the deterministic translation system"""
    
    # Example that should use deterministic rules
    deterministic_example = {
        "type": "FunctionDefinition",
        "name": "square_evens",
        "parameters": [{"name": "xs"}],
        "body": {
            "type": "Application",
            "function": {"type": "Variable", "name": "filter"},
            "arguments": [
                {
                    "type": "Lambda",
                    "parameters": [{"name": "x"}],
                    "body": {
                        "type": "BinaryOp",
                        "operator": "%",
                        "left": {"type": "Variable", "name": "x"},
                        "right": {"type": "Literal", "value": 2, "literalType": "Integer"}
                    }
                },
                {"type": "Variable", "name": "xs"}
            ]
        }
    }
    
    system = GuaranteedTranslationSystem()
    result = system.translate_with_guarantees(json.dumps(deterministic_example))
    
    print("=== Deterministic Translation Test ===")
    print(f"Code:\n{result['python_code']}\n")
    print(f"Deterministic: {result['is_deterministic']}")
    print(f"Method: {result['method_used']}")
    print(f"Confidence: {result['confidence']}")
    print()
    
    # Check stats
    stats = system.get_system_stats()
    print(f"System Stats: {stats}")


if __name__ == "__main__":
    test_deterministic_translation()
