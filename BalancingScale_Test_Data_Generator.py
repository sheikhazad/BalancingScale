import random
from typing import List, Tuple, Dict, Optional

class ScaleNode:
    def __init__(self, name: str):
        self.name = name
        self.left_mass = 0
        self.right_mass = 0
        self.left_child: Optional['ScaleNode'] = None
        self.right_child: Optional['ScaleNode'] = None
        self.total_mass = 1  # Each scale weighs 1kg itself
        self.adjust_left = 0
        self.adjust_right = 0

def generate_random_tree(num_scales: int) -> Tuple[List[ScaleNode], ScaleNode]:
    """Generate a random valid scale tree with the specified number of scales"""
    if num_scales < 1:
        raise ValueError("Number of scales must be at least 1")
    
    scales = [ScaleNode(f"S{i+1}") for i in range(num_scales)]
    root = scales[0]
    
    # Connect scales in tree structure (no cycles)
    for i in range(num_scales - 1):
        current = scales[i]
        if random.random() < 0.7 and current.left_child is None:  # 70% chance left child
            current.left_child = scales[i+1]
        elif current.right_child is None:  # Else try right child
            current.right_child = scales[i+1]
        # If both sides already have children, we'll skip adding to this node
    
    # Add random masses (1-10kg) to empty pans
    for scale in scales:
        if scale.left_child is None and random.random() < 0.8:  # 80% chance for mass
            scale.left_mass = random.randint(1, 10)
        if scale.right_child is None and random.random() < 0.8:
            scale.right_mass = random.randint(1, 10)
    
    return scales, root

def calculate_balance(node: ScaleNode) -> int:
    """Recursively calculate balance adjustments and total mass for a scale"""
    if node is None:
        return 0
    
    # Calculate total mass for left and right subtrees
    left_total = node.left_mass + calculate_balance(node.left_child)
    right_total = node.right_mass + calculate_balance(node.right_child)
    
    # Calculate needed adjustments to balance
    node.adjust_left = max(0, right_total - left_total)
    node.adjust_right = max(0, left_total - right_total)
    
    # Total mass includes: self (1kg) + left + right + adjustments
    node.total_mass = 1 + left_total + right_total + node.adjust_left + node.adjust_right
    
    return node.total_mass

def generate_test_case(num_scales: int) -> Tuple[List[str], List[str]]:
    """Generate complete test case with input and expected output"""
    scales, root = generate_random_tree(num_scales)
    calculate_balance(root)
    
    input_lines = []
    expected_output = []
    
    for scale in scales:
        # Format: ScaleName, LeftValue, RightValue
        left = str(scale.left_mass) if scale.left_child is None else scale.left_child.name
        right = str(scale.right_mass) if scale.right_child is None else scale.right_child.name
        input_lines.append(f"{scale.name},{left},{right}")
        
        # Format: ScaleName, LeftAdjustment, RightAdjustment
        expected_output.append(f"{scale.name},{scale.adjust_left},{scale.adjust_right}")
    
    return input_lines, expected_output

def write_test_files(base_name: str, input_lines: List[str], expected_lines: List[str]):
    """Write test case to input and expected output files"""
    with open(f"{base_name}_input.txt", "w") as f:
        f.write("\n".join(input_lines))
    with open(f"{base_name}_expected.txt", "w") as f:
        f.write("\n".join(expected_lines))
    print(f"\nGenerated test files: {base_name}_input.txt and {base_name}_expected.txt")

def get_user_input() -> int:
    """Prompt user for number of scales and validate input"""
    while True:
        try:
            num_scales = int(input("How many scales would you like to generate? (1-20): "))
            if 1 <= num_scales <= 20:
                return num_scales
            print("Please enter a number between 1 and 20")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("=== Balancing Scale Test Case Generator ===")
    print("This tool generates random test cases for the scale balancing problem.")
    print("Each test case includes an input file and expected output file.\n")
    
    random.seed()  # Seed with current time for different results each run
    
    num_scales = get_user_input()
    test_name = input("Enter a name for this test case (e.g., 'small', 'large'): ").strip() or f"scale{num_scales}"
    
    input_lines, expected_output = generate_test_case(num_scales)
    write_test_files(test_name, input_lines, expected_output)
    
    print("\nTest case preview:")
    print("\nInput:")
    print("\n".join(input_lines[:5]))  # Show first 5 lines if large
    if len(input_lines) > 5:
        print(f"... ({len(input_lines)-5} more lines)")
    
    print("\nExpected Output:")
    print("\n".join(expected_output[:5]))
    if len(expected_output) > 5:
        print(f"... ({len(expected_output)-5} more lines)")

if __name__ == "__main__":
    main()