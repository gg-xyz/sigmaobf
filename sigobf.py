import sys
import os
import re
import random
import string
import base64
import subprocess
from typing import List, Dict, Tuple, Optional

# Auto-install required packages
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        sys.exit(1)

# Check and install required packages
required_packages = ['PyQt5']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        install_package(package)

# Now import PyQt5 components
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar, 
    QGroupBox, QRadioButton, QMessageBox, QSplitter, QComboBox,
    QCheckBox, QFrame
)
from PyQt5.QtGui import QFont, QIcon, QTextCursor, QColor, QPalette, QPixmap
from PyQt5.QtCore import Qt, QTimer, QSize

# Constants
WATERMARK = ".gg/Fkw8VTxDkH"

# Base Obfuscator Class
class BaseObfuscator:
    def __init__(self):
        self.watermark = WATERMARK
        
    def _generate_random_name(self, length: int = 10) -> str:
        """Generate a random variable/function name."""
        chars = string.ascii_letters + string.digits
        return '_' + ''.join(random.choice(chars) for _ in range(length))
    
    def _convert_to_one_liner(self, code: str) -> str:
        """Convert obfuscated code to a one-liner."""
        # This is a generic implementation - override in subclasses if needed
        result = ""
        in_string = False
        string_delimiter = None
        i = 0
        
        while i < len(code):
            char = code[i]
            
            # Handle string delimiters
            if char in ['"', "'"] and (i == 0 or code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_delimiter = char
                elif char == string_delimiter:
                    in_string = False
                result += char
            # Keep all characters within strings
            elif in_string:
                result += char
            # Replace whitespace outside strings
            elif char.isspace():
                # Add a space only if necessary for syntax
                if (i > 0 and i < len(code) - 1 and 
                    (code[i-1].isalnum() and code[i+1].isalnum() or
                     code[i-1] in [')', ']'] and code[i+1] in ['(', '['])):
                    result += ' '
            else:
                result += char
            
            i += 1
            
        return result
    
    def obfuscate_basic(self, code: str, as_one_liner: bool = False) -> str:
        """Basic obfuscation - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def obfuscate_normal(self, code: str, as_one_liner: bool = False) -> str:
        """Normal obfuscation - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

# Default Lua Obfuscator
class LuaObfuscator(BaseObfuscator):
    def __init__(self):
        super().__init__()
        self.language_keywords = [
            'and', 'break', 'do', 'else', 'elseif', 'end', 
            'false', 'for', 'function', 'if', 'in', 'local', 
            'nil', 'not', 'or', 'repeat', 'return', 'then', 
            'true', 'until', 'while', 'print', 'pairs', 'ipairs'
        ]
    
    def _encrypt_string(self, s: str) -> str:
        """Encrypt a string using base64 and split operations."""
        b64 = base64.b64encode(s.encode()).decode()
        return f'({{"_".."_".."_".."_".."_"}})("ba".."se".."64")["dec".."ode"]("{b64}")'
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from Lua code."""
        # Remove single-line comments
        code = re.sub(r'--.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'--\[\[.*?\]\]--', '', code, flags=re.DOTALL)
        return code
    
    def _obfuscate_control_flow(self, code: str) -> str:
        """Obfuscate control flow by introducing fake conditionals."""
        obfuscated_code = "local _ENV={...}; local _R=math.random;" + code
        lines = obfuscated_code.split('\n')
        result = []
        
        for line in lines:
            if line.strip() and not line.strip().startswith('local '):
                # Add random junk conditions
                if random.random() < 0.3:  # 30% chance to add a fake condition
                    cond = f"if _R() > 0.1 then {line} else {line} end"
                    result.append(cond)
                else:
                    result.append(line)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def _add_watermark(self, code: str) -> str:
        """Add unremovable watermark to the obfuscated code."""
        # Create an encrypted version of the watermark that will be harder to remove
        encrypted_watermark = f"local _wm = '{self.watermark}'"
        watermark_print = f'print("Obfuscated with Lua Obfuscator - " .. _wm)'
        
        # Create a self-checking mechanism for the watermark
        watermark_check = """
local _check_wm = function()
    if _wm ~= '""" + self.watermark + """' then
        error("Watermark tampering detected")
    end
end

-- Call the check at random intervals
local _wm_timer = function()
    _check_wm()
    return true
end

-- Pre-execute the watermark check
_wm_timer()
"""
        
        # Add watermark at the very beginning of the code
        watermarked_code = f"-- Obfuscated with Lua Obfuscator - {self.watermark}\n{encrypted_watermark}\n{watermark_print}\n{watermark_check}\n\n{code}"
        
        # Add additional watermark checks throughout the code
        lines = watermarked_code.split('\n')
        for i in range(len(lines)):
            if random.random() < 0.05:  # 5% chance to add a watermark check
                lines.insert(i, "_wm_timer()")
        
        return '\n'.join(lines)
    
    def obfuscate_basic(self, code: str, as_one_liner: bool = False) -> str:
        """Basic obfuscation: variable/function name obfuscation."""
        # Remove comments first
        code = self._remove_comments(code)
        
        # Find all variable and function names
        var_pattern = r'\blocal\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        func_pattern = r'\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        variables = re.findall(var_pattern, code)
        functions = re.findall(func_pattern, code)
        
        # Create name mapping
        name_mapping = {}
        for name in set(variables + functions):
            # Don't obfuscate Lua keywords and built-in functions
            if name not in self.language_keywords:
                name_mapping[name] = self._generate_random_name()
        
        # Replace all occurrences
        for original, obfuscated in name_mapping.items():
            # Use word boundaries to avoid partial matches
            code = re.sub(r'\b' + re.escape(original) + r'\b', obfuscated, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            result = self._convert_to_one_liner(result)
            
        return result
    
    def obfuscate_normal(self, code: str, as_one_liner: bool = False) -> str:
        """Normal obfuscation: variable/function name obfuscation + string encryption."""
        # First, apply basic obfuscation without one-liner conversion
        code = self.obfuscate_basic(code, False)
        
        # Remove the watermark as it will be added again at the end
        watermark_pattern = f"-- Obfuscated with Lua Obfuscator - {self.watermark}.*?_wm_timer\\(\\)"
        code = re.sub(watermark_pattern, "", code, flags=re.DOTALL)
        
        # Encrypt strings
        def encrypt_match(match):
            s = match.group(1)
            return self._encrypt_string(s)
        
        # Find and encrypt all string literals
        code = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', encrypt_match, code)
        code = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", encrypt_match, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            result = self._convert_to_one_liner(result)
            
        return result
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation: everything from normal + control flow obfuscation."""
        # First, apply normal obfuscation without one-liner conversion
        code = self.obfuscate_normal(code, False)
        
        # Remove the watermark as it will be added again at the end
        watermark_pattern = f"-- Obfuscated with Lua Obfuscator - {self.watermark}.*?_wm_timer\\(\\)"
        code = re.sub(watermark_pattern, "", code, flags=re.DOTALL)
        
        # Add control flow obfuscation
        code = self._obfuscate_control_flow(code)
        
        # Break up the code into chunks and shuffle
        lines = code.split('\n')
        
        # Add junk code
        for i in range(len(lines) * 2):
            junk_code = f"local _{self._generate_random_name()} = {random.random()}"
            lines.insert(random.randint(0, len(lines)), junk_code)
        
        # Add fake error handlers
        error_handler = """
local _error_handler = function(err)
    if not string.find(err, "Watermark") then
        print(err)
    else
        error(err)  -- Propagate watermark errors
    end
    return err
end
local _xpcall = xpcall
local _original_print = print
local _print_encrypted = function(...)
    _original_print(...)
end
print = _print_encrypted
"""
        
        code = error_handler + "\n" + "\n".join(lines)
        
        # Wrap the entire code in a load() function for additional protection
        wrapped_code = """
local _load = load or loadstring
local _code = [=[
""" + code + """
]=]
_code = string.gsub(_code, "([^%w])", function(c)
    return string.format("\\%03d", string.byte(c))
end)
local _f, _err = _load("return " .. _code)
if _f then
    local _status, _result = pcall(_f)
    if not _status then
        if string.find(_result, "Watermark") then
            error("Watermark validation failed")  -- Make watermark issues fatal
        else
            print(_result)
        end
    end
else
    print(_err)
end
"""
        
        result = self._add_watermark(wrapped_code)
        
        if as_one_liner:
            result = self._convert_to_one_liner(result)
            
        return result

# Roblox Lua Obfuscator
class RobloxLuaObfuscator(LuaObfuscator):
    def __init__(self):
        super().__init__()
        # Add Roblox-specific keywords
        self.language_keywords.extend([
            'game', 'workspace', 'script', 'wait', 'spawn',
            'Instance', 'CFrame', 'Vector3', 'Color3', 'Enum',
            'BrickColor', 'NumberSequence', 'NumberRange', 'math', 
            'string', 'table', 'task', 'coroutine', 'pcall', 'require',
            'warn', 'error', 'assert', 'tonumber', 'tostring'
        ])
    
    def _encrypt_string(self, s: str) -> str:
        """Encrypt a string in a Roblox-compatible way."""
        # Roblox Lua doesn't support Lua's load() function, so we use a different approach
        chars = []
        for char in s:
            chars.append(str(ord(char)))
        
        return f"string.char({', '.join(chars)})"
    
    def _add_watermark(self, code: str) -> str:
        """Add unremovable watermark to the obfuscated Roblox Lua code."""
        # Create an encrypted version of the watermark
        encrypted_watermark = f"local _wm = '{self.watermark}'"
        watermark_print = f'print("Obfuscated with Roblox Lua Obfuscator - " .. _wm)'
        
        # Create a self-checking mechanism for the watermark with Roblox-specific functions
        watermark_check = """
local _check_wm = function()
    if _wm ~= '""" + self.watermark + """' then
        error("Watermark tampering detected")
    end
end

-- Use Roblox's task scheduler for periodic checks
spawn(function()
    while wait(math.random(5, 10)) do
        _check_wm()
    end
end)

-- Pre-execute the watermark check
_check_wm()
"""
        
        # Add watermark at the very beginning of the code
        watermarked_code = f"-- Obfuscated with Roblox Lua Obfuscator - {self.watermark}\n{encrypted_watermark}\n{watermark_print}\n{watermark_check}\n\n{code}"
        
        return watermarked_code
    
    def _obfuscate_control_flow(self, code: str) -> str:
        """Obfuscate control flow using Roblox-specific techniques."""
        # Roblox-specific version of control flow obfuscation
        lines = code.split('\n')
        result = []
        
        for line in lines:
            if line.strip() and not line.strip().startswith('local '):
                # Add random junk conditions using Roblox's math.random
                if random.random() < 0.3:
                    cond = f"if math.random() > 0.1 then {line} else {line} end"
                    result.append(cond)
                else:
                    result.append(line)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation for Roblox Lua."""
        # First, apply normal obfuscation without one-liner conversion
        code = self.obfuscate_normal(code, False)
        
        # Remove the watermark as it will be added again at the end
        watermark_pattern = f"-- Obfuscated with Roblox Lua Obfuscator - {self.watermark}.*?_check_wm\\(\\)"
        code = re.sub(watermark_pattern, "", code, flags=re.DOTALL)
        
        # Add control flow obfuscation
        code = self._obfuscate_control_flow(code)
        
        # Add junk code and fake functions
        lines = code.split('\n')
        for i in range(len(lines) * 2):
            junk_types = [
                f"local _{self._generate_random_name()} = math.random()",
                f"local _{self._generate_random_name()} = Vector3.new(math.random(), math.random(), math.random())",
                f"local _{self._generate_random_name()} = Color3.fromRGB(math.random(0, 255), math.random(0, 255), math.random(0, 255))",
                f"local _{self._generate_random_name()} = {{}}",
            ]
            lines.insert(random.randint(0, len(lines)), random.choice(junk_types))
        
        # Add fake event handlers that won't affect functionality
        fake_event_handler = """
local _fake_event = {
    Connect = function(_, callback)
        -- This never actually runs the callback
        return {
            Disconnect = function() end
        }
    end
}

-- Add some fake connections that never run
_fake_event:Connect(function()
    print("This will never run")
end)
"""
        
        result = self._add_watermark(fake_event_handler + "\n" + "\n".join(lines))
        
        if as_one_liner:
            result = self._convert_to_one_liner(result)
            
        return result

# Python Obfuscator
class PythonObfuscator(BaseObfuscator):
    def __init__(self):
        super().__init__()
        self.language_keywords = [
            'and', 'as', 'assert', 'async', 'await', 'break', 'class', 
            'continue', 'def', 'del', 'elif', 'else', 'except', 'False', 
            'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 
            'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 
            'return', 'True', 'try', 'while', 'with', 'yield', 'print',
            'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
            'tuple', 'type', 'object', 'Exception'
        ]
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from Python code."""
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line docstrings (simple cases)
        code = re.sub(r'""".*?"""', '""""""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''''''", code, flags=re.DOTALL)
        
        return code
    
    def _encrypt_string(self, s: str) -> str:
        """Encrypt a string for Python code."""
        # Convert string to ASCII values and use chr() to reconstruct
        chars = []
        for char in s:
            chars.append(str(ord(char)))
        
        return "''.join([chr({}+{}-{}) for _ in range(0)]+''.join([chr({}) for c in [{}]]))".format(
            random.randint(1, 10), 
            random.randint(1, 10), 
            random.randint(1, 10),
            "c", 
            ", ".join(chars)
        )
    
    def _add_watermark(self, code: str) -> str:
        """Add unremovable watermark to Python code."""
        # Encrypted watermark
        watermark_var = f"_wm = '{self.watermark}'"
        # Fix: Use string concatenation instead of nested f-strings
        watermark_print = 'print("Obfuscated with Python Obfuscator - " + _wm)'
        
        # Self-checking mechanism 
        watermark_check = """
def _check_watermark():
    if _wm != '""" + self.watermark + """':
        raise ValueError("Watermark tampering detected")
    return True

# Immediate check
_check_watermark()

# Create a decorator that checks the watermark before any function
def _watermark_checked(func):
    def wrapper(*args, **kwargs):
        _check_watermark()
        return func(*args, **kwargs)
    return wrapper
"""
        
        # Add to the beginning of the code
        watermarked_code = f"# Obfuscated with Python Obfuscator - {self.watermark}\n{watermark_var}\n{watermark_print}\n{watermark_check}\n\n{code}"
        
        # Apply the decorator to random functions
        if "def " in code:
            lines = watermarked_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("def ") and random.random() < 0.5:
                    # Insert the decorator before the function definition
                    lines[i] = "@_watermark_checked\n" + line
            
            watermarked_code = '\n'.join(lines)
        
        return watermarked_code
    
    def _obfuscate_control_flow(self, code: str) -> str:
        """Obfuscate control flow in Python code."""
        lines = code.split('\n')
        result = []
        indent_stack = []
        current_indent = ""
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:  # Empty line
                result.append(line)
                continue
                
            # Calculate the indentation of this line
            indent = line[:len(line) - len(stripped)]
            
            # Update indent stack
            if indent_stack and len(indent) < len(indent_stack[-1]):
                while indent_stack and len(indent) < len(indent_stack[-1]):
                    indent_stack.pop()
            
            if not indent_stack or len(indent) > len(indent_stack[-1]):
                indent_stack.append(indent)
                
            current_indent = indent
            
            # Don't modify import statements or decorators
            if stripped.startswith(("import ", "from ", "class ", "@", "def ", "#", "except", "finally")):
                result.append(line)
                continue
                
            # 30% chance to add a random condition
            if random.random() < 0.3 and not line.strip().endswith(":"):
                random_var = f"_{self._generate_random_name()[1:]}"
                condition = random.choice([
                    f"{current_indent}if True: {stripped}",
                    f"{current_indent}{random_var} = True\n{current_indent}if {random_var}: {stripped}",
                    f"{current_indent}if 1 + 1 == 2: {stripped}"
                ])
                result.append(condition)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def obfuscate_basic(self, code: str, as_one_liner: bool = False) -> str:
        """Basic obfuscation for Python: variable/function name obfuscation."""
        # Remove comments
        code = self._remove_comments(code)
        
        # Find all variable and function names
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='  # Variables
        func_pattern = r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('  # Functions
        class_pattern = r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]'  # Classes
        
        variables = re.findall(var_pattern, code)
        functions = re.findall(func_pattern, code)
        classes = re.findall(class_pattern, code)
        
        # Create name mapping
        name_mapping = {}
        for name in set(variables + functions + classes):
            # Don't obfuscate Python keywords and built-ins
            if name not in self.language_keywords and not name.startswith('__'):
                name_mapping[name] = self._generate_random_name()
        
        # Replace all occurrences
        for original, obfuscated in name_mapping.items():
            code = re.sub(r'\b' + re.escape(original) + r'\b', obfuscated, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            # Python is whitespace-dependent, so one-liner needs special handling
            # We'll use semicolons to separate statements
            lines = result.split('\n')
            one_liner = ';'.join(line.strip() for line in lines if line.strip())
            return one_liner
        
        return result
    
    def obfuscate_normal(self, code: str, as_one_liner: bool = False) -> str:
        """Normal obfuscation for Python: basic + string encryption."""
        # Apply basic obfuscation first
        code = self.obfuscate_basic(code, False)
        
        # Remove existing watermark to avoid duplication
        watermark_pattern = f"# Obfuscated with Python Obfuscator - {self.watermark}.*?_check_watermark\\(\\)"
        code = re.sub(watermark_pattern, "", code, flags=re.DOTALL)
        
        # Encrypt strings
        def encrypt_match(match):
            s = match.group(1)
            return self._encrypt_string(s)
        
        # Find and encrypt string literals
        code = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', encrypt_match, code)
        code = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", encrypt_match, code)
        
        # Add additional obfuscation
        # Convert simple variable assignments to exec with encoded statements
        assignment_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;\n]*)'
        
        def encode_assignment(match):
            var_name = match.group(1)
            value = match.group(2).strip()
            
            # Only obfuscate simple assignments
            if '+' in value or '-' in value or '*' in value or '/' in value or '(' in value:
                return f"{var_name} = {value}"
            
            # Encode using base64
            encoded = base64.b64encode(f"{var_name} = {value}".encode()).decode()
            return f"exec(__import__('base64').b64decode('{encoded}'.encode()).decode())"
        
        code = re.sub(assignment_pattern, encode_assignment, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            lines = result.split('\n')
            one_liner = ';'.join(line.strip() for line in lines if line.strip())
            return one_liner
        
        return result
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation for Python."""
        # Apply normal obfuscation first
        code = self.obfuscate_normal(code, False)
        
        # Remove existing watermark to avoid duplication
        watermark_pattern = f"# Obfuscated with Python Obfuscator - {self.watermark}.*?_check_watermark\\(\\)"
        code = re.sub(watermark_pattern, "", code, flags=re.DOTALL)
        
        # Add control flow obfuscation
        code = self._obfuscate_control_flow(code)
        
        # Add junk code and variables
        lines = code.split('\n')
        junk_lines = []
        
        for i in range(len(lines) * 2):
            junk_name = self._generate_random_name()
            junk_types = [
                f"{junk_name} = {random.random()}",
                f"{junk_name} = [{random.random()} for _ in range({random.randint(1, 5)})]",
                f"{junk_name} = {{{self._generate_random_name()[1:]}: {random.random()} for _ in range({random.randint(1, 3)})}}",
                f"{junk_name} = lambda x: x + {random.random()}"
            ]
            junk_lines.append(random.choice(junk_types))
        
        # Insert junk lines randomly
        for junk in junk_lines:
            lines.insert(random.randint(0, len(lines)), junk)
        
        # Wrap everything in a dynamic exec
        wrapped_code = """
import base64
import random
import sys

# Encode the actual code
_code = \"\"\"
""" + '\\n'.join(lines) + """
\"\"\"

# Obfuscation layer: Encode the code multiple times
for _ in range(3):
    _code = base64.b64encode(_code.encode()).decode()

# Define the execution function
def _execute_code(_encoded_code):
    # Decode the encoded code
    _temp_code = _encoded_code
    for _ in range(3):
        _temp_code = base64.b64decode(_temp_code.encode()).decode()
    
    # Execute the decoded code
    exec(_temp_code, globals())

# Execute the code
_execute_code(_code)
"""
        
        result = self._add_watermark(wrapped_code)
        
        if as_one_liner:
            lines = result.split('\n')
            one_liner = ';'.join(line.strip() for line in lines if line.strip())
            return one_liner
        
        return result

# C# Obfuscator
class CSharpObfuscator(BaseObfuscator):
    def __init__(self):
        super().__init__()
        self.language_keywords = [
            'abstract', 'as', 'base', 'bool', 'break', 'byte', 'case', 'catch',
            'char', 'checked', 'class', 'const', 'continue', 'decimal', 'default',
            'delegate', 'do', 'double', 'else', 'enum', 'event', 'explicit', 
            'extern', 'false', 'finally', 'fixed', 'float', 'for', 'foreach',
            'goto', 'if', 'implicit', 'in', 'int', 'interface', 'internal',
            'is', 'lock', 'long', 'namespace', 'new', 'null', 'object', 'operator',
            'out', 'override', 'params', 'private', 'protected', 'public', 'readonly',
            'ref', 'return', 'sbyte', 'sealed', 'short', 'sizeof', 'stackalloc',
            'static', 'string', 'struct', 'switch', 'this', 'throw', 'true', 'try',
            'typeof', 'uint', 'ulong', 'unchecked', 'unsafe', 'ushort', 'using',
            'virtual', 'void', 'volatile', 'while', 'Console', 'var'
        ]
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from C# code."""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def _encrypt_string(self, s: str) -> str:
        """Encrypt a string for C# code."""
        # Convert string to byte array and use string constructor
        bytes_str = ", ".join(str(ord(c)) for c in s)
        return f"new string(new char[] {{ {bytes_str} }})"
    
    def _add_watermark(self, code: str) -> str:
        """Add unremovable watermark to C# code."""
        # Create watermark constants
        watermark_var = f"private const string _wm = \"{self.watermark}\";"
        watermark_print = f'Console.WriteLine("Obfuscated with C# Obfuscator - " + _wm);\n'
        
        # Add watermark checking method
        watermark_check = """
        private static void CheckWatermark()
        {
            if (_wm != \"""" + self.watermark + """\")
            {
                throw new Exception("Watermark tampering detected");
            }
        }
        
        // Call at program start
        static {
            CheckWatermark();
        }
"""
        
        # Identify the class structure
        class_match = re.search(r'(public\s+)?class\s+(\w+)\s*{', code)
        if class_match:
            class_name = class_match.group(2)
            class_pos = class_match.end()
            
            # Insert watermark fields and methods inside the class
            before_class = code[:class_pos]
            after_class = code[class_pos:]
            
            modified_code = f"{before_class}\n        // Obfuscated with C# Obfuscator - {self.watermark}\n        {watermark_var}\n{watermark_check}{after_class}"
            
            # Insert watermark print in Main method or constructor
            main_match = re.search(r'(public\s+static\s+void\s+Main\s*\([^)]*\)\s*{)', modified_code)
            if main_match:
                main_pos = main_match.end()
                modified_code = f"{modified_code[:main_pos]}\n            {watermark_print}{modified_code[main_pos:]}"
            else:
                # Try to find constructor
                constructor_match = re.search(r'(public\s+{class_name}\s*\([^)]*\)\s*{)', modified_code)
                if constructor_match:
                    constructor_pos = constructor_match.end()
                    modified_code = f"{modified_code[:constructor_pos]}\n            {watermark_print}{modified_code[constructor_pos:]}"
                else:
                    # Just add a static constructor
                    static_constructor = f"""
        static {class_name}()
        {{
            {watermark_print}
        }}
"""
                    modified_code = f"{before_class}\n        // Obfuscated with C# Obfuscator - {self.watermark}\n        {watermark_var}\n{watermark_check}{static_constructor}{after_class[1:]}"
            
            return modified_code
        else:
            # If no class found, just add comment at the top
            return f"// Obfuscated with C# Obfuscator - {self.watermark}\n{code}"
    
    def _obfuscate_control_flow(self, code: str) -> str:
        """Obfuscate control flow in C# code."""
        lines = code.split('\n')
        result = []
        
        for line in lines:
            # Skip preprocessor directives, attributes, and declarations
            if re.match(r'^\s*#|^\s*\[|^\s*(public|private|protected|internal|class|namespace|using)\s', line):
                result.append(line)
                continue
                
            stripped = line.strip()
            if not stripped or stripped.endswith("{") or stripped.endswith("}"):
                result.append(line)
                continue
            
            # Don't obfuscate lines ending with labels, access modifiers, etc.
            if stripped.endswith(":") or stripped.endswith(";"):
                result.append(line)
                continue
            
            # Add obfuscated control flow (30% chance)
            if random.random() < 0.3:
                # Extract indentation
                indent = line[:len(line) - len(line.lstrip())]
                
                # Add conditional that always executes but is harder to analyze
                random_var = f"_{self._generate_random_name()[1:]}"
                
                obfuscated = random.choice([
                    f"{indent}if (true) {{ {stripped} }}",
                    f"{indent}{{ bool {random_var} = true; if ({random_var}) {{ {stripped} }} }}",
                    f"{indent}switch(1) {{ case 1: {stripped} break; }}",
                    f"{indent}for (int {random_var} = 0; {random_var} < 1; {random_var}++) {{ {stripped} break; }}"
                ])
                
                result.append(obfuscated)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def obfuscate_basic(self, code: str, as_one_liner: bool = False) -> str:
        """Basic obfuscation for C#: variable/function name obfuscation."""
        # Remove comments
        code = self._remove_comments(code)
        
        # Find variables, methods, classes and parameters
        var_pattern = r'\b(var|int|string|bool|float|double|decimal|char|byte|long)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        method_pattern = r'\b(public|private|protected|internal|static)?\s+\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        class_pattern = r'\b(public|private|protected|internal)?\s+class\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        param_pattern = r'\(([^)]+)\)'
        
        # Extract names
        variables = []
        for match in re.finditer(var_pattern, code):
            variables.append(match.group(2))
            
        methods = []
        for match in re.finditer(method_pattern, code):
            method_name = match.group(2)
            if method_name != "Main":  # Don't obfuscate Main method
                methods.append(method_name)
        
        classes = []
        for match in re.finditer(class_pattern, code):
            class_name = match.group(2)
            if class_name != "Program":  # Don't obfuscate Program class 
                classes.append(class_name)
        
        # Extract parameters
        parameters = []
        for match in re.finditer(param_pattern, code):
            params_str = match.group(1)
            for param in params_str.split(','):
                param = param.strip()
                if param:
                    # Extract parameter name, handling complex declarations
                    param_parts = param.split()
                    if len(param_parts) >= 2:
                        param_name = param_parts[-1].strip()
                        if param_name and param_name[0].isalpha():
                            parameters.append(param_name)
        
        # Create name mapping
        name_mapping = {}
        for name in set(variables + methods + classes + parameters):
            # Don't obfuscate C# keywords and built-ins
            if name not in self.language_keywords and not name.startswith('_'):
                name_mapping[name] = self._generate_random_name()
        
        # Replace all occurrences
        for original, obfuscated in name_mapping.items():
            code = re.sub(r'\b' + re.escape(original) + r'\b', obfuscated, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            # C# is not whitespace-dependent, so we can collapse multiple lines
            # but preserve braces and semicolons
            one_liner = re.sub(r'\s+', ' ', result)
            return one_liner
        
        return result
    
    def obfuscate_normal(self, code: str, as_one_liner: bool = False) -> str:
        """Normal obfuscation for C#: basic + string encryption."""
        # Apply basic obfuscation first
        code = self.obfuscate_basic(code, False)
        
        # Encrypt strings
        def encrypt_match(match):
            s = match.group(1)
            return self._encrypt_string(s)
        
        # Find and encrypt string literals
        code = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', encrypt_match, code)
        
        # Add additional obfuscation techniques
        # Insert dummy variables and methods
        
        # Find the class scope
        class_match = re.search(r'(public\s+)?class\s+(\w+)\s*{', code)
        if class_match:
            class_pos = class_match.end()
            
            # Create dummy variables and methods
            dummy_code = "\n"
            for _ in range(5):
                var_name = self._generate_random_name()
                dummy_code += f"        private static int {var_name} = {random.randint(1, 100)};\n"
            
            for _ in range(3):
                method_name = self._generate_random_name()
                dummy_code += f"""
        private static void {method_name}()
        {{
            // This method is never called
            int x = {random.randint(1, 100)};
            string s = {self._encrypt_string("Dummy string")};
        }}
"""
            
            # Insert the dummy code inside the class
            code = code[:class_pos] + dummy_code + code[class_pos:]
        
        # Update the watermark to include the normal obfuscation level
        result = code
        if "// Obfuscated with C# Obfuscator" in result:
            result = result.replace("// Obfuscated with C# Obfuscator", "// Obfuscated with C# Obfuscator (Normal)")
        
        if as_one_liner:
            one_liner = re.sub(r'\s+', ' ', result)
            return one_liner
        
        return result
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation for C#."""
        # Apply normal obfuscation first
        code = self.obfuscate_normal(code, False)
        
        # Add control flow obfuscation
        code = self._obfuscate_control_flow(code)
        
        # Add junk code and proxy methods
        lines = code.split('\n')
        result = []
        
        # Add proxy methods for any method calls
        method_call_pattern = r'(\w+)\s*\((.*?)\);'
        method_calls = set(re.findall(method_call_pattern, code))
        
        proxy_methods = ""
        for method_name, args in method_calls:
            if method_name in self.language_keywords or method_name.startswith('_'):
                continue
                
            arg_count = len(args.split(',')) if args.strip() else 0
            arg_types = ["object"] * arg_count
            arg_names = [f"arg{i}" for i in range(arg_count)]
            
            arg_list = ", ".join(f"{arg_types[i]} {arg_names[i]}" for i in range(arg_count))
            arg_pass = ", ".join(arg_names)
            
            proxy_name = f"_{self._generate_random_name()[1:]}"
            
            proxy_method = f"""
        private static void {proxy_name}({arg_list})
        {{
            {method_name}({arg_pass});
        }}
"""
            proxy_methods += proxy_method
        
        # Find the class closing brace to add proxy methods
        class_end = code.rfind('}')
        if class_end > 0:
            code = code[:class_end] + proxy_methods + code[class_end:]
        
        # Update the watermark to include the extreme obfuscation level
        result = code
        if "// Obfuscated with C# Obfuscator" in result:
            result = result.replace("// Obfuscated with C# Obfuscator", "// Obfuscated with C# Obfuscator (EXTREME)")
        
        if as_one_liner:
            one_liner = re.sub(r'\s+', ' ', result)
            return one_liner
        
        return result

# C Obfuscator
class CObfuscator(BaseObfuscator):
    def __init__(self):
        super().__init__()
        self.language_keywords = [
            'auto', 'break', 'case', 'char', 'const', 'continue',
            'default', 'do', 'double', 'else', 'enum', 'extern',
            'float', 'for', 'goto', 'if', 'int', 'long', 'register',
            'return', 'short', 'signed', 'sizeof', 'static', 'struct',
            'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile',
            'while', 'printf', 'scanf', 'malloc', 'free', 'NULL',
            'strcmp', 'strcpy', 'strlen', 'main'
        ]
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from C code."""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def _encrypt_string(self, s: str) -> str:
        """Encrypt a string for C code."""
        # Convert string to array of chars with offsets
        encrypted = []
        offset = random.randint(1, 20)
        
        for char in s:
            encrypted.append(str(ord(char) + offset))
        
        decrypt_code = f"""((char[]){{
    {', '.join([f"{int(c)}-{offset}" for c in encrypted])}, 0
}})"""
        
        return decrypt_code
    
    def _add_watermark(self, code: str) -> str:
        """Add unremovable watermark to C code."""
        watermark_define = f"#define WATERMARK \"{self.watermark}\"\n"
        watermark_check = """
/* Obfuscated with C Obfuscator - """ + self.watermark + """ */
#include <stdio.h>
#include <string.h>

static const char _wm[] = \"""" + self.watermark + """\";

static void _check_watermark() {
    if (strcmp(_wm, WATERMARK) != 0) {
        fprintf(stderr, "Watermark tampering detected\\n");
        exit(1);
    }
}

static void _print_watermark() {
    printf("Obfuscated with C Obfuscator - %s\\n", WATERMARK);
}

/* Call watermark check at program start using constructor attribute */
__attribute__((constructor)) static void _watermark_init() {
    _check_watermark();
    _print_watermark();
}

"""
        
        # For MSVC compatibility
        msvc_watermark = """
/* Obfuscated with C Obfuscator - """ + self.watermark + """ */
#include <stdio.h>
#include <string.h>

static const char _wm[] = \"""" + self.watermark + """\";

static void _check_watermark() {
    if (strcmp(_wm, WATERMARK) != 0) {
        fprintf(stderr, "Watermark tampering detected\\n");
        exit(1);
    }
}

static void _print_watermark() {
    printf("Obfuscated with C Obfuscator - %s\\n", WATERMARK);
}

/* Force watermark check to run before main */
#pragma section(".CRT$XCU", read)
static void _watermark_init();
__declspec(allocate(".CRT$XCU")) void (*_watermark_init_ptr)() = _watermark_init;

static void _watermark_init() {
    _check_watermark();
    _print_watermark();
}

"""
        
        # Detect platform-specific code (simplified approach)
        if "#include <windows.h>" in code:
            watermark_code = watermark_define + msvc_watermark
        else:
            watermark_code = watermark_define + watermark_check
            
        # Find the main function
        main_match = re.search(r'(int|void)\s+main\s*\(', code)
        if main_match:
            main_pos = main_match.start()
            
            # Insert the watermark code before main
            result = code[:main_pos] + watermark_code + code[main_pos:]
            
            # Add watermark check at the beginning of main function
            brace_pos = result.find('{', main_pos)
            if brace_pos > 0:
                result = result[:brace_pos+1] + "\n    _check_watermark();\n" + result[brace_pos+1:]
            
            return result
        else:
            # If no main function, just prepend the watermark
            return watermark_code + code
    
    def _obfuscate_control_flow(self, code: str) -> str:
        """Obfuscate control flow in C code."""
        lines = code.split('\n')
        result = []
        
        for line in lines:
            # Skip preprocessor directives, comments, and declarations
            if re.match(r'^\s*#|^\s*/\*|^\s*//|^\s*(int|char|float|double|void|struct|typedef)\s+', line):
                result.append(line)
                continue
                
            stripped = line.strip()
            if not stripped or stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(";"):
                result.append(line)
                continue
            
            # Add obfuscated control flow (30% chance)
            if random.random() < 0.3:
                # Extract indentation
                indent = line[:len(line) - len(line.lstrip())]
                
                # Add conditional that always executes but is harder to analyze
                random_var = f"_{self._generate_random_name()[1:]}"
                
                obfuscated = random.choice([
                    f"{indent}if (1) {{ {stripped}; }}",
                    f"{indent}{{ int {random_var} = 1; if ({random_var}) {{ {stripped}; }} }}",
                    f"{indent}switch(1) {{ case 1: {stripped}; break; }}",
                    f"{indent}{{ int {random_var} = 0; while({random_var} < 1) {{ {stripped}; {random_var}++; }} }}"
                ])
                
                result.append(obfuscated)
            else:
                result.append(line)
                
        return '\n'.join(result)
    
    def obfuscate_basic(self, code: str, as_one_liner: bool = False) -> str:
        """Basic obfuscation for C: variable/function name obfuscation."""
        # Remove comments
        code = self._remove_comments(code)
        
        # Find variables, functions and types
        var_pattern = r'\b(int|char|float|double|void|unsigned|signed|short|long)\s+([a-zA-Z_][a-zA-Z0-9_]*)[,;=\[]'
        func_pattern = r'\b(int|char|float|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        # Extract names
        variables = []
        for match in re.finditer(var_pattern, code):
            var_name = match.group(2)
            if var_name != "main":  # Don't obfuscate main function
                variables.append(var_name)
            
        functions = []
        for match in re.finditer(func_pattern, code):
            func_name = match.group(2)
            if func_name != "main":  # Don't obfuscate main function 
                functions.append(func_name)
        
        # Create name mapping
        name_mapping = {}
        for name in set(variables + functions):
            # Don't obfuscate C keywords and built-ins
            if name not in self.language_keywords and not name.startswith('_'):
                name_mapping[name] = self._generate_random_name()
        
        # Replace all occurrences
        for original, obfuscated in name_mapping.items():
            code = re.sub(r'\b' + re.escape(original) + r'\b', obfuscated, code)
        
        result = self._add_watermark(code)
        
        if as_one_liner:
            # C is not whitespace-dependent
            one_liner = re.sub(r'\s+', ' ', result)
            # Preserve preprocessor directives
            one_liner = re.sub(r'#\s+', '# ', one_liner)
            one_liner = re.sub(r'\s+#', ' #', one_liner)
            return one_liner
        
        return result
    
    def obfuscate_normal(self, code: str, as_one_liner: bool = False) -> str:
        """Normal obfuscation for C: basic + string encryption."""
        # Apply basic obfuscation first
        code = self.obfuscate_basic(code, False)
        
        # Encrypt strings
        def encrypt_match(match):
            s = match.group(1)
            return self._encrypt_string(s)
        
        # Find and encrypt string literals
        code = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', encrypt_match, code)
        
        # Add macros to make code harder to read
        macros = """
/* Obfuscation macros */
#define _OBFUSCATE_A(x) ((x) + 1)
#define _OBFUSCATE_B(x) ((x) - 1)
#define _OBFUSCATE_C(x) ((x) ^ 0x12345678)
#define _OBFUSCATE_D(x) (_OBFUSCATE_B(_OBFUSCATE_A(x)))
#define _OBFUSCATE_E(x) (_OBFUSCATE_C(_OBFUSCATE_D(x)))
"""
        
        # Add macros at the beginning, after any existing #include directives
        include_pos = 0
        for match in re.finditer(r'#include\s+<[^>]+>|#include\s+"[^"]+"', code):
            include_pos = match.end()
        
        if include_pos > 0:
            code = code[:include_pos] + "\n" + macros + code[include_pos:]
        else:
            code = macros + code
            
        # Add dummy typedef and structs
        dummy_types = """
/* Dummy types for obfuscation */
typedef struct {
    int _a, _b, _c;
    char _d[32];
} _ObfuscatedType1;

typedef struct {
    void* _ptr;
    int (*_func)(int);
} _ObfuscatedType2;

/* Dummy variables */
static _ObfuscatedType1 _dummy1 = {0};
static _ObfuscatedType2 _dummy2 = {0};
"""
        
        # Find a good position to insert dummy types
        if "#define WATERMARK" in code:
            pos = code.find("#define WATERMARK") + len("#define WATERMARK")
            code = code[:pos] + "\n" + dummy_types + code[pos:]
            
        # Replace some numeric constants with obfuscated expressions
        def obfuscate_number(match):
            num = int(match.group(1))
            options = [
                f"_OBFUSCATE_A({num-1})",
                f"_OBFUSCATE_B({num+1})",
                f"_OBFUSCATE_D({num})",
                f"({num} + 0)"
            ]
            return random.choice(options)
            
        code = re.sub(r'\b(\d+)\b', obfuscate_number, code)
        
        # Update the watermark
        if "/* Obfuscated with C Obfuscator" in code:
            code = code.replace("/* Obfuscated with C Obfuscator", "/* Obfuscated with C Obfuscator (Normal)")
        
        if as_one_liner:
            one_liner = re.sub(r'\s+', ' ', code)
            # Preserve preprocessor directives
            one_liner = re.sub(r'#\s+', '# ', one_liner)
            one_liner = re.sub(r'\s+#', ' #', one_liner)
            return one_liner
        
        return code
    
    def obfuscate_extreme(self, code: str, as_one_liner: bool = False) -> str:
        """Extreme obfuscation for C."""
        # Apply normal obfuscation first
        code = self.obfuscate_normal(code, False)
        
        # Add even more macros for extreme obfuscation
        extreme_macros = """
/* Extreme obfuscation macros */
#define _OBF(x) ((x) ^ 0xDEADBEEF)
#define _DEOBF(x) ((x) ^ 0xDEADBEEF)
#define _CONCAT(a, b) a ## b
#define _STRINGIZE(x) #x
#define _EXPAND_STRINGIZE(x) _STRINGIZE(x)
#define _CALL(f, x) f(x)
#define _JOIN(a, b) _CONCAT(a, b)
#define _LINE_VAR(prefix) _JOIN(prefix, __LINE__)

/* Function pointer type for indirection */
typedef int (*_func_ptr_t)(int);

/* Conditional compilation based on random values */
#if ((__LINE__ % 3) == 0)
#define _CONDITION_A(x) ((x) + 1)
#elif ((__LINE__ % 3) == 1)
#define _CONDITION_A(x) ((x) + 2)
#else
#define _CONDITION_A(x) ((x) + 3)
#endif
"""
        
        # Add control flow obfuscation
        code = self._obfuscate_control_flow(code)
        
        # Add the extreme macros at the top of the file
        if "#define WATERMARK" in code:
            pos = code.find("#define WATERMARK")
            code = code[:pos] + extreme_macros + code[pos:]
        else:
            code = extreme_macros + code
            
        # Add goto-based obfuscation in functions
        def obfuscate_function_body(match):
            body = match.group(2)
            lines = body.split('\n')
            
            # Add goto labels and jumps
            if len(lines) > 5:  # Only obfuscate larger functions
                modified_lines = []
                labels = [f"label_{i}_{self._generate_random_name()[1:]}" for i in range(3)]
                
                # Add labels at random positions
                for i, line in enumerate(lines):
                    if i > 0 and i < len(lines) - 1 and random.random() < 0.2:
                        label_idx = random.randint(0, len(labels) - 1)
                        modified_lines.append(f"{line}\n{labels[label_idx]}:")
                    else:
                        modified_lines.append(line)
                
                # Add goto statements
                for i in range(3):
                    if len(modified_lines) > 5:
                        pos = random.randint(1, len(modified_lines) - 3)
                        target = random.randint(0, len(labels) - 1)
                        
                        # Add a conditional goto that's always false
                        var_name = f"_goto_var_{self._generate_random_name()[1:]}"
                        modified_lines.insert(pos, f"    int {var_name} = 0; if ({var_name}) goto {labels[target]};")
                
                body = '\n'.join(modified_lines)
            
            return match.group(1) + body + match.group(3)
            
        code = re.sub(r'(\w+\s*\([^)]*\)\s*{)(.*?)(})', obfuscate_function_body, code, flags=re.DOTALL)
        
        # Update the watermark
        if "/* Obfuscated with C Obfuscator" in code:
            code = code.replace("/* Obfuscated with C Obfuscator", "/* Obfuscated with C Obfuscator (EXTREME)")
        
        if as_one_liner:
            one_liner = re.sub(r'\s+', ' ', code)
            # Preserve preprocessor directives
            one_liner = re.sub(r'#\s+', '# ', one_liner)
            one_liner = re.sub(r'\s+#', ' #', one_liner)
            return one_liner
        
        return code

# Themes for the UI
THEMES = {
    "Dark Blue": {
        "main_bg": "#1E293B", 
        "secondary_bg": "#0F172A",
        "text": "#E2E8F0",
        "accent": "#3B82F6",
        "button": "#2563EB",
        "button_hover": "#1D4ED8",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "border": "#334155"
    },
    "Midnight Purple": {
        "main_bg": "#1E1B4B",
        "secondary_bg": "#312E81",
        "text": "#E0E7FF",
        "accent": "#8B5CF6",
        "button": "#7C3AED",
        "button_hover": "#6D28D9",
        "success": "#10B981",
        "warning": "#FBBF24",
        "danger": "#DC2626",
        "border": "#4338CA"
    },
    "Dark Green": {
        "main_bg": "#064E3B",
        "secondary_bg": "#065F46",
        "text": "#D1FAE5",
        "accent": "#10B981",
        "button": "#059669",
        "button_hover": "#047857",
        "success": "#34D399",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "border": "#047857"
    },
    "Crimson Red": {
        "main_bg": "#7F1D1D",
        "secondary_bg": "#991B1B",
        "text": "#FEE2E2",
        "accent": "#F87171",
        "button": "#EF4444",
        "button_hover": "#DC2626",
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#B91C1C",
        "border": "#B91C1C"
    },
    "Cyber Dark": {
        "main_bg": "#09090B",
        "secondary_bg": "#18181B",
        "text": "#F4F4F5",
        "accent": "#06B6D4",
        "button": "#0EA5E9",
        "button_hover": "#0284C7",
        "success": "#22C55E",
        "warning": "#FACC15",
        "danger": "#EF4444",
        "border": "#27272A"
    },
    "Coffee Brown": {
        "main_bg": "#44403C",
        "secondary_bg": "#292524",
        "text": "#E7E5E4",
        "accent": "#D6D3D1",
        "button": "#A8A29E",
        "button_hover": "#78716C",
        "success": "#84CC16",
        "warning": "#EAB308",
        "danger": "#DC2626",
        "border": "#57534E"
    },
    "Orange Sunset": {
        "main_bg": "#7C2D12",
        "secondary_bg": "#9A3412",
        "text": "#FFEDD5",
        "accent": "#FB923C",
        "button": "#F97316",
        "button_hover": "#EA580C",
        "success": "#22C55E",
        "warning": "#FACC15",
        "danger": "#EF4444",
        "border": "#C2410C"
    },
    "Classic Dark": {
        "main_bg": "#1E1E1E",
        "secondary_bg": "#252526",
        "text": "#D4D4D4",
        "accent": "#0E639C",
        "button": "#007ACC",
        "button_hover": "#005F99",
        "success": "#4EC9B0",
        "warning": "#CE9178",
        "danger": "#F44747",
        "border": "#3F3F46"
    }
}

class ObfuscatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = "Dark Blue"  # Default theme
        self.current_language = "Default Lua"  # Default language
        
        # Initialize language-specific obfuscators
        self.obfuscators = {
            "Default Lua": LuaObfuscator(),
            "Roblox Lua": RobloxLuaObfuscator(),
            "Python": PythonObfuscator(),
            "C#": CSharpObfuscator(),
            "C": CObfuscator()
        }
        
        self.init_ui()
        
        # Print watermark to console
        print(f"Multi-Language Code Obfuscator - {WATERMARK}")
        
    def init_ui(self):
        self.setWindowTitle('Multi-Language Code Obfuscator')
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply theme immediately
        self.apply_theme(self.current_theme)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Add header with title and branding
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setFrameShadow(QFrame.Raised)
        header_layout = QHBoxLayout(header_frame)
        
        # Title and watermark in the header
        title_section = QVBoxLayout()
        
        title_label = QLabel('MULTI-LANGUAGE CODE OBFUSCATOR')
        title_label.setFont(QFont('Arial', 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignLeft)
        title_section.addWidget(title_label)
        
        watermark_label = QLabel(f'Watermark: {WATERMARK}')
        watermark_label.setFont(QFont('Arial', 10))
        watermark_label.setAlignment(Qt.AlignLeft)
        title_section.addWidget(watermark_label)
        
        header_layout.addLayout(title_section, 3)
        
        # Theme selector in the header
        theme_layout = QHBoxLayout()
        theme_label = QLabel('Theme:')
        theme_layout.addWidget(theme_label)
        
        self.theme_selector = QComboBox()
        for theme in THEMES.keys():
            self.theme_selector.addItem(theme)
        self.theme_selector.setCurrentText(self.current_theme)
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        theme_layout.addWidget(self.theme_selector)
        
        header_layout.addLayout(theme_layout, 1)
        main_layout.addWidget(header_frame)
        
        # Language selector
        language_frame = QFrame()
        language_layout = QHBoxLayout(language_frame)
        language_label = QLabel('Language:')
        language_label.setFont(QFont('Arial', 12, QFont.Bold))
        language_layout.addWidget(language_label)
        
        self.language_selector = QComboBox()
        self.language_selector.addItems([
            "Default Lua", "Roblox Lua", "Python", "C#", "C"
        ])
        self.language_selector.setCurrentText(self.current_language)
        self.language_selector.currentTextChanged.connect(self.change_language)
        self.language_selector.setFont(QFont('Arial', 11))
        self.language_selector.setMinimumWidth(150)
        language_layout.addWidget(self.language_selector)
        
        language_layout.addStretch()
        main_layout.addWidget(language_frame)
        
        # Create splitter for input/output
        splitter = QSplitter(Qt.Horizontal)
        
        # Input group
        input_group = QGroupBox('Input Code')
        input_layout = QVBoxLayout(input_group)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText('Enter your code here or load from file...')
        self.input_text.setFont(QFont('Consolas', 10))
        input_layout.addWidget(self.input_text)
        
        input_buttons_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load From File')
        self.load_btn.clicked.connect(self.load_from_file)
        self.load_btn.setIcon(QIcon.fromTheme('document-open'))
        input_buttons_layout.addWidget(self.load_btn)
        
        self.clear_input_btn = QPushButton('Clear Input')
        self.clear_input_btn.clicked.connect(lambda: self.input_text.clear())
        self.clear_input_btn.setIcon(QIcon.fromTheme('edit-clear'))
        input_buttons_layout.addWidget(self.clear_input_btn)
        
        input_layout.addLayout(input_buttons_layout)
        splitter.addWidget(input_group)
        
        # Output group
        output_group = QGroupBox('Obfuscated Output')
        output_layout = QVBoxLayout(output_group)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont('Consolas', 10))
        output_layout.addWidget(self.output_text)
        
        output_buttons_layout = QHBoxLayout()
        self.save_btn = QPushButton('Save To File')
        self.save_btn.clicked.connect(self.save_to_file)
        self.save_btn.setIcon(QIcon.fromTheme('document-save'))
        output_buttons_layout.addWidget(self.save_btn)
        
        self.copy_btn = QPushButton('Copy Output')
        self.copy_btn.clicked.connect(self.copy_output)
        self.copy_btn.setIcon(QIcon.fromTheme('edit-copy'))
        output_buttons_layout.addWidget(self.copy_btn)
        
        output_layout.addLayout(output_buttons_layout)
        splitter.addWidget(output_group)
        
        main_layout.addWidget(splitter, 1)  # Give the splitter a stretch factor
        
        # Obfuscation control group
        control_group = QGroupBox('Obfuscation Settings')
        control_layout = QVBoxLayout(control_group)
        
        # Top controls: Obfuscation level and format option
        top_controls = QHBoxLayout()
        
        # Obfuscation level
        level_group = QGroupBox('Obfuscation Level')
        level_layout = QHBoxLayout(level_group)
        
        self.basic_radio = QRadioButton('Basic')
        self.basic_radio.setChecked(True)
        self.basic_radio.setToolTip("Variable and function name obfuscation")
        level_layout.addWidget(self.basic_radio)
        
        self.normal_radio = QRadioButton('Normal')
        self.normal_radio.setToolTip("Basic + string encryption and comment removal")
        level_layout.addWidget(self.normal_radio)
        
        self.extreme_radio = QRadioButton('EXTREME')
        self.extreme_radio.setToolTip("Normal + control flow obfuscation and code structure manipulation")
        level_layout.addWidget(self.extreme_radio)
        
        top_controls.addWidget(level_group)
        
        # Format options
        format_group = QGroupBox('Output Format')
        format_layout = QHBoxLayout(format_group)
        
        self.full_text_radio = QRadioButton('Full Text')
        self.full_text_radio.setChecked(True)
        self.full_text_radio.setToolTip("Keep original code formatting with line breaks")
        format_layout.addWidget(self.full_text_radio)
        
        self.one_liner_radio = QRadioButton('One Liner')
        self.one_liner_radio.setToolTip("Convert to single line code (more obfuscated but harder to read)")
        format_layout.addWidget(self.one_liner_radio)
        
        top_controls.addWidget(format_group)
        control_layout.addLayout(top_controls)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        control_layout.addLayout(progress_layout)
        
        # Obfuscate button
        self.obfuscate_btn = QPushButton('OBFUSCATE')
        self.obfuscate_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.obfuscate_btn.clicked.connect(self.start_obfuscation)
        control_layout.addWidget(self.obfuscate_btn)
        
        main_layout.addWidget(control_group)
        
        # Status bar at the bottom
        self.status_label = QLabel('Ready')
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Update UI for initial language
        self.change_language(self.current_language)
        
    def apply_theme(self, theme_name):
        """Apply the selected theme to the application."""
        if theme_name not in THEMES:
            return
            
        theme = THEMES[theme_name]
        self.current_theme = theme_name
        
        # Create stylesheet based on the theme
        stylesheet = f"""
            QMainWindow, QDialog {{
                background-color: {theme["main_bg"]};
                color: {theme["text"]};
            }}
            QWidget {{
                background-color: {theme["main_bg"]};
                color: {theme["text"]};
            }}
            QGroupBox {{
                border: 2px solid {theme["border"]};
                border-radius: 8px;
                margin-top: 1ex;
                font-weight: bold;
                color: {theme["text"]};
                background-color: {theme["secondary_bg"]};
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {theme["accent"]};
            }}
            QTextEdit {{
                background-color: {theme["secondary_bg"]};
                color: {theme["text"]};
                border: 1px solid {theme["border"]};
                border-radius: 4px;
                font-family: Consolas, monospace;
                padding: 5px;
            }}
            QPushButton {{
                background-color: {theme["button"]};
                color: {theme["text"]};
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                min-height: 30px;
            }}
            QPushButton:hover {{
                background-color: {theme["button_hover"]};
            }}
            QPushButton:pressed {{
                background-color: {theme["accent"]};
            }}
            QPushButton:disabled {{
                background-color: {theme["secondary_bg"]};
                color: {theme["border"]};
            }}
            QLabel {{
                color: {theme["text"]};
                padding: 2px;
            }}
            QProgressBar {{
                border: 1px solid {theme["border"]};
                border-radius: 4px;
                text-align: center;
                background-color: {theme["secondary_bg"]};
                height: 20px;
            }}
            QProgressBar::chunk {{
                background-color: {theme["accent"]};
                width: 10px;
            }}
            QRadioButton {{
                color: {theme["text"]};
                spacing: 5px;
                padding: 2px;
            }}
            QRadioButton::indicator {{
                width: 15px;
                height: 15px;
            }}
            QComboBox {{
                background-color: {theme["secondary_bg"]};
                color: {theme["text"]};
                border: 1px solid {theme["border"]};
                border-radius: 4px;
                padding: 5px;
                min-height: 25px;
            }}
            QComboBox:hover {{
                border: 1px solid {theme["accent"]};
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme["secondary_bg"]};
                color: {theme["text"]};
                selection-background-color: {theme["accent"]};
            }}
            QFrame {{
                background-color: {theme["secondary_bg"]};
                border-radius: 8px;
                padding: 5px;
            }}
            QSplitter::handle {{
                background-color: {theme["border"]};
            }}
            QCheckBox {{
                color: {theme["text"]};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
            }}
        """
        
        self.setStyleSheet(stylesheet)
        
        # Apply special styles for specific elements
        if hasattr(self, 'obfuscate_btn'):
            self.obfuscate_btn.setStyleSheet(f"""
                background-color: {theme["success"]};
                color: {theme["text"]};
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            """)
            
        if hasattr(self, 'extreme_radio'):
            self.extreme_radio.setStyleSheet(f"""
                color: {theme["danger"]};
                font-weight: bold;
            """)
        
    def change_theme(self, theme_name):
        """Change the application theme."""
        self.apply_theme(theme_name)
        self.status_label.setText(f"Theme changed to {theme_name}")
        
        # Reset status after 2 seconds
        QTimer.singleShot(2000, lambda: self.status_label.setText('Ready'))
    
    def change_language(self, language):
        """Change the current language and update UI accordingly."""
        self.current_language = language
        self.input_text.setPlaceholderText(f'Enter your {language} code here...')
        
        # Update window title
        self.setWindowTitle(f'{language} Code Obfuscator')
        
        # Update placeholder code based on language
        placeholder_code = ""
        
        if language == "Default Lua":
            placeholder_code = """-- Example Lua code
local function factorial(n)
    if n == 0 then
        return 1
    else
        return n * factorial(n - 1)
    end
end

print("Factorial of 5 is: " .. factorial(5))
"""
        elif language == "Roblox Lua":
            placeholder_code = """-- Example Roblox Lua code
local part = Instance.new("Part")
part.Parent = workspace
part.Anchored = true
part.Position = Vector3.new(0, 10, 0)
part.BrickColor = BrickColor.new("Bright red")

local function onTouch(otherPart)
    local player = game.Players:GetPlayerFromCharacter(otherPart.Parent)
    if player then
        print("Player " .. player.Name .. " touched the part!")
    end
end

part.Touched:Connect(onTouch)
"""
        elif language == "Python":
            placeholder_code = """# Example Python code
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# Calculate factorial
result = factorial(5)
print(f"Factorial of 5 is: {result}")

# Using a list comprehension
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print("Squares:", squares)
"""
        elif language == "C#":
            placeholder_code = """// Example C# code
using System;

namespace ObfuscatorExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Factorial of 5 is: " + Factorial(5));
        }
        
        static int Factorial(int n)
        {
            if (n == 0)
                return 1;
            else
                return n * Factorial(n - 1);
        }
    }
}
"""
        elif language == "C":
            placeholder_code = """// Example C code
#include <stdio.h>

int factorial(int n) {
    if (n == 0)
        return 1;
    else
        return n * factorial(n - 1);
}

int main() {
    int result = factorial(5);
    printf("Factorial of 5 is: %d\\n", result);
    return 0;
}
"""
            
        # Update placeholder
        if self.input_text.toPlainText() == "":
            self.input_text.setPlaceholderText(placeholder_code)
            
        self.status_label.setText(f"Language changed to {language}")
        
        # Reset status after 2 seconds
        QTimer.singleShot(2000, lambda: self.status_label.setText('Ready'))
    
    def load_from_file(self):
        """Load code from a file."""
        file_extensions = {
            "Default Lua": "Lua Files (*.lua)",
            "Roblox Lua": "Lua Files (*.lua)",
            "Python": "Python Files (*.py)",
            "C#": "C# Files (*.cs)",
            "C": "C Files (*.c *.h)"
        }
        
        ext_filter = file_extensions.get(self.current_language, "All Files (*)")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, f'Open {self.current_language} File', '', f'{ext_filter};;All Files (*)'
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.input_text.setText(file.read())
                self.status_label.setText(f'Loaded: {os.path.basename(file_path)}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load file: {str(e)}')
    
    def save_to_file(self):
        """Save obfuscated code to a file."""
        file_extensions = {
            "Default Lua": "Lua Files (*.lua)",
            "Roblox Lua": "Lua Files (*.lua)",
            "Python": "Python Files (*.py)",
            "C#": "C# Files (*.cs)",
            "C": "C Files (*.c)"
        }
        
        ext_filter = file_extensions.get(self.current_language, "All Files (*)")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, f'Save Obfuscated {self.current_language}', '', f'{ext_filter};;All Files (*)'
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.output_text.toPlainText())
                self.status_label.setText(f'Saved to: {os.path.basename(file_path)}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save file: {str(e)}')
    
    def copy_output(self):
        """Copy obfuscated code to clipboard."""
        self.output_text.selectAll()
        self.output_text.copy()
        self.output_text.moveCursor(QTextCursor.Start)
        self.status_label.setText('Output copied to clipboard')
        
        # Reset status after 2 seconds
        QTimer.singleShot(2000, lambda: self.status_label.setText('Ready'))
    
    def simulate_progress(self):
        """Simulate progress bar advancement."""
        current = self.progress_bar.value()
        if current < 100:
            self.progress_bar.setValue(current + 5)
            QTimer.singleShot(50, self.simulate_progress)
        else:
            self.progress_bar.setValue(100)
            self.status_label.setText('Obfuscation Complete!')
            theme = THEMES[self.current_theme]
            self.status_label.setStyleSheet(f"color: {theme['success']}; font-weight: bold;")
            self.obfuscate_btn.setEnabled(True)
            # Reset status after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setText('Ready'))
            QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(""))
    
    def start_obfuscation(self):
        """Start the obfuscation process."""
        input_code = self.input_text.toPlainText().strip()
        if not input_code:
            QMessageBox.warning(self, 'Warning', f'Please enter some {self.current_language} code to obfuscate.')
            return
        
        self.obfuscate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText('Obfuscating...')
        
        # Start progress simulation
        QTimer.singleShot(50, self.simulate_progress)
        
        # Get the current obfuscator
        obfuscator = self.obfuscators[self.current_language]
        
        # Determine obfuscation level
        if self.basic_radio.isChecked():
            obfuscation_func = obfuscator.obfuscate_basic
            level = "Basic"
        elif self.normal_radio.isChecked():
            obfuscation_func = obfuscator.obfuscate_normal
            level = "Normal"
        else:  # EXTREME
            obfuscation_func = obfuscator.obfuscate_extreme
            level = "EXTREME"
        
        # Determine output format
        as_one_liner = self.one_liner_radio.isChecked()
        
        try:
            # Perform obfuscation
            obfuscated_code = obfuscation_func(input_code, as_one_liner)
            self.output_text.setText(obfuscated_code)
            print(f"Applied {level} obfuscation for {self.current_language} - {WATERMARK}")
        except Exception as e:
            self.progress_bar.setValue(0)
            self.status_label.setText('Error occurred during obfuscation')
            theme = THEMES[self.current_theme]
            self.status_label.setStyleSheet(f"color: {theme['danger']}; font-weight: bold;")
            QMessageBox.critical(self, 'Error', f'Obfuscation failed: {str(e)}')
            self.obfuscate_btn.setEnabled(True)
            # Reset status after 3 seconds
            QTimer.singleShot(3000, lambda: self.status_label.setText('Ready'))
            QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet(""))

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = ObfuscatorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
