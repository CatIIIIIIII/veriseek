import re

def extract_param(src_code):
    # Updated regex pattern to capture the entire parameter definition
    pattern = r'\bparameter\s+(\w+)\s*=\s*([^;]+);'
    matches = re.findall(pattern, src_code)

    # Create a dictionary to store parameter names and their values
    params = {name: value.strip() for name, value in matches}

    return params

# Example usage
src_code = """
module example;
    parameter INIT = 0;
    parameter WIDTH = 8;
    // Some comment
    parameter DELAY = 10;
endmodule
"""

params = extract_param(src_code)
print(params)
