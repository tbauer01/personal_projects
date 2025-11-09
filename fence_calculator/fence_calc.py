from ortools.linear_solver import pywraplp
import math

def solve_cutting_plan(pieces, num_boards, board_length, min_leftover, board_type, show_header=True):
    """Reusable solver function for cutting optimization."""
    
    num_pieces = len(pieces)
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Solver not available.")

    # Variables: x[i][j] = 1 if piece i assigned to board j
    x = {}
    for i in range(num_pieces):
        for j in range(num_boards):
            x[i, j] = solver.BoolVar(f"x_{i}_{j}")

    # Leftover variables for each board
    leftover = [solver.NumVar(0, board_length, f"leftover_{j}") for j in range(num_boards)]

    # Constraint: each piece assigned to exactly one board
    for i in range(num_pieces):
        solver.Add(solver.Sum([x[i, j] for j in range(num_boards)]) == 1)

    # Constraint: sum of pieces on each board + leftover = board_length
    for j in range(num_boards):
        solver.Add(solver.Sum([pieces[i] * x[i, j] for i in range(num_pieces)]) + leftover[j] == board_length)

    # Constraint: each board must have at least min_leftover inches leftover
    for j in range(num_boards):
        solver.Add(leftover[j] >= min_leftover)

    # Objective: find any feasible solution (minimize total leftover to use boards efficiently)
    solver.Minimize(solver.Sum(leftover))

    # Solve
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        if show_header:
            print(f"Solution found for {board_type}! All boards have at least {min_leftover} inches leftover.\n")
        total_leftover = 0
        min_leftover_found = float('inf')
        
        cutting_plan = []
        for j in range(num_boards):
            board_pieces = [pieces[i] for i in range(num_pieces) if x[i, j].solution_value() > 0.5]
            leftover_value = round(leftover[j].solution_value())  # Round to remove floating point precision issues
            total_leftover += leftover_value
            min_leftover_found = min(min_leftover_found, leftover_value)
            
            pieces_sum = sum(board_pieces)
            cutting_plan.append({
                'board': j+1,
                'pieces': board_pieces,
                'total': pieces_sum,
                'leftover': leftover_value
            })
            print(f"Board {j+1}: pieces = {board_pieces} (total: {pieces_sum}), leftover = {leftover_value}")
        
        print(f"\nSummary:")
        print(f"Total leftover across all boards: {round(total_leftover)} inches")
        print(f"Minimum leftover per board: {round(min_leftover_found)} inches")
        print(f"Average leftover per board: {round(total_leftover/num_boards, 1)} inches")
        
        return {
            'success': True,
            'num_boards': num_boards,
            'board_length': board_length,
            'board_type': board_type,
            'cutting_plan': cutting_plan,
            'total_leftover': total_leftover
        }
    else:
        print(f"No solution found for {board_type} that guarantees at least {min_leftover} inches leftover per board.")
        print("Try reducing the minimum leftover requirement or increasing the number of boards.")
        return {
            'success': False,
            'num_boards': num_boards,
            'board_length': board_length,
            'board_type': board_type
        }

def two_by_four():
    """Calculate optimal 2x4 fence board cutting plan."""
    
    # Board specifications
    board_length = 192  # 16 feet in inches
    num_boards = 8
    min_leftover = 8
    board_type = "2x4 boards"
    
    # Piece specifications
    section_1 = [40, 40]
    section_2 = [57, 57]
    section_3 = [80, 80]
    section_4 = [47, 47] 
    section_5 = [78, 78] 
    section_6 = [78, 78] 
    section_7 = [78, 78] 
    gate_1 = [42, 42, 30, 30]
    gate_2 = [60, 60, 36, 36]
    
    # Calculate diagonal braces with Pythagorean theorem + 10% safety margin, rounded up
    gate_1_diagonal = math.ceil((gate_1[0]**2 + gate_1[2]**2)**0.5 * 1.1)
    gate_2_diagonal = math.ceil((gate_2[0]**2 + gate_2[2]**2)**0.5 * 1.1)
    
    gate_1.append(gate_1_diagonal)
    gate_2.append(gate_2_diagonal)

    pieces = section_1 + section_2 + section_3 + section_4 + section_5 + section_6 + section_7 + gate_1 + gate_2
    
    return solve_cutting_plan(pieces, num_boards, board_length, min_leftover, board_type, show_header=False)


def five_quarters_by_six():
    """Calculate optimal 5/4" x 6" fence board cutting plan."""
    
    board_length = 24*12
    num_boards = 2
    min_leftover = 6
    board_type = "5/4\" x 6\" boards"
    
    section_1 = 40
    section_2 = 57
    section_3 = 80
    section_4 = 47
    section_5_6 = 78 * 3
    section_6_7 = 0
    gate_1 = 42
    gate_2 = 60

    # Create pieces list - single pieces for 5/4" x 6" boards
    pieces = [section_1, section_2, section_3, section_4, section_5_6, section_6_7, gate_1, gate_2]
    
    return solve_cutting_plan(pieces, num_boards, board_length, min_leftover, board_type, show_header=False)

def gate_panels(): 
    """Calculate cutting plan for gate panel slats."""
    # Gate dimensions and slat counts
    g1_height = 30
    g1_width = 42  # Gate 1 width
    g1_one_by_four_count = 7
    g1_one_by_six_count = 0
    # gap 0.68 inches
    
    g2_height = 36
    g2_width = 60  # Gate 2 width
    g2_one_by_four_count = 7
    g2_one_by_six_count = 0
    # gap 0.45 inches

    board_length = 192
    num_boards = 4
    min_leftover = 4
    board_type = "1x4 gate panels"
    
    # Create pieces list for gate panels
    # Gate 1: 7 pieces of gate width each
    gate_1_pieces = [g1_width] * g1_one_by_four_count
    
    # Gate 2: 7 pieces of gate width each  
    gate_2_pieces = [g2_width] * g2_one_by_four_count
    
    # Combine all gate panel pieces
    pieces = gate_1_pieces + gate_2_pieces
    
    print(f"Gate 1 panels: {g1_one_by_four_count} pieces at {g1_width}\" each")
    print(f"Gate 2 panels: {g2_one_by_four_count} pieces at {g2_width}\" each")
    print(f"Total gate panel pieces: {len(pieces)}")
    print()
    
    return solve_cutting_plan(pieces, num_boards, board_length, min_leftover, board_type, show_header=False)

def four_by_four_posts():
    """Calculate cutting plan for 4x4 fence posts."""
    print("Boards 1 - 5: 10 pieces at 60 inches each")
    return {
        'board_type': '4x4 posts',
        'num_boards': 5,
        'board_length': 120,  # 10 feet in inches
        'description': '10 5-foot posts from 5 10-footers'
    }

def two_by_two():
    """Calculate cutting plan for 2x2 sections."""
    print("Boards 1 - 2: 4 pieces at 30 inches each")
    return {
        'board_type': '2x2 sections', 
        'num_boards': 2,
        'board_length': 96,  # 8 feet in inches
        'description': '4 30-inch sections from 2 8-footers'
    }
    

def main():
    """Main function that triggers the fence calculation."""
    print("=== Fence Board Cutting Calculator ===")
    print("Calculating optimal cutting plans...\n")
    
    print("--- 2x4 Board Calculation ---")
    result_2x4 = two_by_four()
    
    print("\n" + "="*50 + "\n")
    
    print("--- 5/4\" x 6\" Board Calculation ---")
    result_5x6 = five_quarters_by_six()
    
    print("\n" + "="*50 + "\n")
    
    print("--- Gate Panel Calculation ---")
    result_gate = gate_panels()
    
    print("\n" + "="*50 + "\n")
    
    print("--- 4x4 Post Calculation ---")
    result_4x4 = four_by_four_posts()
    
    print("\n" + "="*50 + "\n")
    
    print("--- 2x2 Calculation ---")
    result_2x2 = two_by_two()
    
    print("\n" + "="*60 + "\n")
    print("LUMBER SHOP SUMMARY")
    print("="*60)
    
    print("\nSUMMARY - LUMBER TO PURCHASE:")
    print(f"• {result_2x4['num_boards']} pieces of 2x4 @ {result_2x4['board_length']//12} feet")
    print(f"• {result_5x6['num_boards']} pieces of 5/4\" x 6\" @ {result_5x6['board_length']//12} feet") 
    print(f"• {result_gate['num_boards']} pieces of 1x4 @ {result_gate['board_length']//12} feet")
    print(f"• {result_4x4['num_boards']} pieces of 4x4 @ {result_4x4['board_length']//12} feet")
    print(f"• {result_2x2['num_boards']} pieces of 2x2 @ {result_2x2['board_length']//12} feet")
    
    print("\nDETAILS - CUT PLANS:")
    print("-" * 40)
    print(f"2x4 BOARDS ({result_2x4['board_length']//12}-foot lengths):")
    if result_2x4['success']:
        for plan in result_2x4['cutting_plan']:
            print(f"  Board {plan['board']}: pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n5/4\" x 6\" BOARDS ({result_5x6['board_length']//12}-foot lengths):")
    if result_5x6['success']:
        for plan in result_5x6['cutting_plan']:
            print(f"  Board {plan['board']}: pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n1x4 BOARDS ({result_gate['board_length']//12}-foot lengths):")
    if result_gate['success']:
        for plan in result_gate['cutting_plan']:
            print(f"  Board {plan['board']}: pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n4x4 POSTS ({result_4x4['board_length']//12}-foot lengths):")
    print(f"  {result_4x4['description']}")
    
    print(f"\n2x2 SECTIONS ({result_2x2['board_length']//12}-foot lengths):")
    print(f"  {result_2x2['description']}")


if __name__ == "__main__":
    main()
