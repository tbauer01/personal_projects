from ortools.linear_solver import pywraplp
import math
import sys

def solve_cutting_plan(pieces, board_lengths, min_leftover, board_type, show_header=True, section_mapping=None, verbose=True):
    """Reusable solver function for cutting optimization with variable board lengths."""
    
    num_pieces = len(pieces)
    num_boards = len(board_lengths)
    
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
    leftover = [solver.NumVar(0, board_lengths[j], f"leftover_{j}") for j in range(num_boards)]

    # Constraint: each piece assigned to exactly one board
    for i in range(num_pieces):
        solver.Add(solver.Sum([x[i, j] for j in range(num_boards)]) == 1)

    # Constraint: sum of pieces on each board + leftover = board_length for that specific board
    for j in range(num_boards):
        solver.Add(solver.Sum([pieces[i] * x[i, j] for i in range(num_pieces)]) + leftover[j] == board_lengths[j])

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
                'board_length': board_lengths[j],
                'pieces': board_pieces,
                'total': pieces_sum,
                'leftover': leftover_value
            })
            if verbose:
                print(f"Board {j+1} ({board_lengths[j]}\"): pieces = {board_pieces} (total: {pieces_sum}), leftover = {leftover_value}")
        
        if verbose:
            print(f"\nSummary:")
            print(f"Total leftover across all boards: {round(total_leftover)} inches")
            print(f"Minimum leftover per board: {round(min_leftover_found)} inches")
            print(f"Average leftover per board: {round(total_leftover/num_boards, 1)} inches")
        
        # Section mapping analysis
        section_boards = {}
        if section_mapping:
            if verbose:
                print(f"\nSection Board Mapping:")
            piece_to_board = {}
            
            # Create mapping of piece index to board number
            for j in range(num_boards):
                for i in range(num_pieces):
                    if x[i, j].solution_value() > 0.5:
                        piece_to_board[i] = j + 1
            
            # Group pieces by section and find which boards are used with their cut lengths
            piece_index = 0
            for section_name, section_pieces in section_mapping.items():
                boards_with_cuts = {}  # board_num -> list of cut lengths
                for piece_length in section_pieces:
                    if piece_index in piece_to_board:
                        board_num = piece_to_board[piece_index]
                        if board_num not in boards_with_cuts:
                            boards_with_cuts[board_num] = []
                        boards_with_cuts[board_num].append(piece_length)
                    piece_index += 1
                
                # Convert to list of tuples (board_num, cut_lengths)
                boards_with_lengths = []
                for board_num in sorted(boards_with_cuts.keys()):
                    cut_lengths = boards_with_cuts[board_num]
                    boards_with_lengths.append((board_num, cut_lengths))
                
                section_boards[section_name] = boards_with_lengths
                if verbose:
                    board_count = len(boards_with_cuts)
                    boards_list = ", ".join([f"#{b}" for b in sorted(boards_with_cuts.keys())])
                    print(f"  {section_name}: {board_count} board(s) - {boards_list}")
        
        return {
            'success': True,
            'num_boards': num_boards,
            'board_lengths': board_lengths,
            'board_type': board_type,
            'cutting_plan': cutting_plan,
            'total_leftover': total_leftover,
            'section_boards': section_boards
        }
    else:
        print(f"No solution found for {board_type} that guarantees at least {min_leftover} inches leftover per board.")
        print("Try reducing the minimum leftover requirement or increasing the number of boards.")
        return {
            'success': False,
            'num_boards': num_boards,
            'board_lengths': board_lengths,
            'board_type': board_type
        }

def two_by_four(verbose=True):
    """Calculate optimal 2x4 fence board cutting plan."""
    
    # Board specifications - array of board lengths you're buying (in inches)
    board_lengths = [192, 192, 192, 192, 192, 192, 192, 192]  # 8 boards at 16 feet each
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
    
    # Create section mapping for tracking board usage per section
    section_mapping = {
        'section_1': section_1,
        'section_2': section_2, 
        'section_3': section_3,
        'section_4': section_4,
        'section_5': section_5,
        'section_6': section_6,
        'section_7': section_7,
        'gate_1': gate_1,
        'gate_2': gate_2
    }
    
    return solve_cutting_plan(pieces, board_lengths, min_leftover, board_type, show_header=False, section_mapping=section_mapping, verbose=verbose)


def five_quarters_by_six(verbose=True):
    """Calculate optimal 5/4" x 6" fence board cutting plan."""
    
    #board_lengths = [24*12, 24*12]  
    board_lengths = [22*12, 16*12, 14*12] 
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
    
    # Create section mapping for tracking board usage per section
    section_mapping = {
        'section_1': [section_1],
        'section_2': [section_2], 
        'section_3': [section_3],
        'section_4': [section_4],
        'section_5_6': [section_5_6],
        'section_6_7': [section_6_7],
        'gate_1': [gate_1],
        'gate_2': [gate_2]
    }
    
    return solve_cutting_plan(pieces, board_lengths, min_leftover, board_type, show_header=False, section_mapping=section_mapping, verbose=verbose)

def gate_panels(verbose=True): 
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

    board_lengths = [192, 192, 192, 192]  # 4 boards at 16 feet each
    min_leftover = 4
    board_type = "1x4 gate panels"
    
    # Create pieces list for gate panels
    # Gate 1: 7 pieces of gate width each
    gate_1_pieces = [g1_width] * g1_one_by_four_count
    
    # Gate 2: 7 pieces of gate width each  
    gate_2_pieces = [g2_width] * g2_one_by_four_count
    
    # Combine all gate panel pieces
    pieces = gate_1_pieces + gate_2_pieces
    
    if verbose:
        print(f"Gate 1 panels: {g1_one_by_four_count} pieces at {g1_width}\" each")
        print(f"Gate 2 panels: {g2_one_by_four_count} pieces at {g2_width}\" each")
        print(f"Total gate panel pieces: {len(pieces)}")
        print()
    
    # Create section mapping for tracking board usage per section
    section_mapping = {
        'gate_1_panels': gate_1_pieces,
        'gate_2_panels': gate_2_pieces
    }
    
    return solve_cutting_plan(pieces, board_lengths, min_leftover, board_type, show_header=False, section_mapping=section_mapping, verbose=verbose)

def four_by_four_posts(verbose=True):
    """Calculate cutting plan for 4x4 fence posts."""
    
    # Board specifications - what you're buying
    board_lengths = [120, 120, 120, 120, 96, 96] 
    min_leftover = 0  # Allow tight cutting for posts
    board_type = "4x4 posts"
    
    # Pieces needed: 8 at 60" and 2 at 96"
    pieces_60 = [60] * 8  # 8 pieces at 60 inches
    pieces_96 = [96] * 2  # 2 pieces at 96 inches
    pieces = pieces_60 + pieces_96
    
    if verbose:
        print(f"Posts needed: {len(pieces_60)} at 60\" and {len(pieces_96)} at 96\"")
        print(f"Total 4x4 pieces: {len(pieces)}")
        print()
    
    # Create section mapping for tracking board usage per section
    section_mapping = {
        '60_inch_posts': pieces_60,
        '96_inch_posts': pieces_96
    }
    
    return solve_cutting_plan(pieces, board_lengths, min_leftover, board_type, show_header=False, section_mapping=section_mapping, verbose=verbose)

def two_by_two(verbose=True):
    """Calculate cutting plan for 2x2 sections."""
    
    # Board specifications - what you're buying  
    board_lengths = [96, 96]  # 2 boards at 8 feet each
    
    # Cutting plan details
    cutting_plan = [
        {'board': 1, 'board_length': 96, 'pieces': [30, 30], 'total': 60, 'leftover': 36},
        {'board': 2, 'board_length': 96, 'pieces': [30, 30], 'total': 60, 'leftover': 36}
    ]
    
    # Print the cutting plan
    if verbose:
        for plan in cutting_plan:
            print(f"Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    
    return {
        'success': True,
        'board_type': '2x2 sections', 
        'num_boards': 2,
        'board_lengths': board_lengths,
        'cutting_plan': cutting_plan,
        'total_leftover': 72
    }
    

def main():
    """Main function that triggers the fence calculation."""
    # Check for verbose flag
    verbose = '-v' in sys.argv
    
    print("=== Fence Board Cutting Calculator ===")
    
    if verbose:
        print("Calculating optimal cutting plans...\n")
        print("--- 2x4 Board Calculation ---")
    result_2x4 = two_by_four(verbose)
    
    if verbose:
        print("\n" + "="*50 + "\n")
        print("--- 5/4\" x 6\" Board Calculation ---")
    result_5x6 = five_quarters_by_six(verbose)
    
    if verbose:
        print("\n" + "="*50 + "\n")
        print("--- Gate Panel Calculation ---")
    result_gate = gate_panels(verbose)
    
    if verbose:
        print("\n" + "="*50 + "\n")
        print("--- 4x4 Post Calculation ---")
    result_4x4 = four_by_four_posts(verbose)
    
    if verbose:
        print("\n" + "="*50 + "\n")
        print("--- 2x2 Calculation ---")
    result_2x2 = two_by_two(verbose)
    
    print("\n" + "="*60 + "\n")
    print("LUMBER SHOP SUMMARY")
    print("="*60)
    
    print("\nSUMMARY - LUMBER TO PURCHASE:")
    
    # 2x4 boards - show unique lengths and counts
    from collections import Counter
    board_counts_2x4 = Counter([length//12 for length in result_2x4['board_lengths']])
    board_summary_2x4 = ", ".join([f"{count}×{length}ft" for length, count in sorted(board_counts_2x4.items())])
    print(f"• 2x4 boards: {board_summary_2x4}")
    
    # 5/4" x 6" boards
    board_counts_5x6 = Counter([length//12 for length in result_5x6['board_lengths']])
    board_summary_5x6 = ", ".join([f"{count}×{length}ft" for length, count in sorted(board_counts_5x6.items())])
    print(f"• 5/4\" x 6\" boards: {board_summary_5x6}")
    
    # 1x4 boards
    board_counts_gate = Counter([length//12 for length in result_gate['board_lengths']])
    board_summary_gate = ", ".join([f"{count}×{length}ft" for length, count in sorted(board_counts_gate.items())])
    print(f"• 1x4 boards: {board_summary_gate}")
    
    # 4x4 boards
    board_counts_4x4 = Counter([length//12 for length in result_4x4['board_lengths']])
    board_summary_4x4 = ", ".join([f"{count}×{length}ft" for length, count in sorted(board_counts_4x4.items())])
    print(f"• 4x4 boards: {board_summary_4x4}")
    
    # 2x2 boards
    board_counts_2x2 = Counter([length//12 for length in result_2x2['board_lengths']])
    board_summary_2x2 = ", ".join([f"{count}×{length}ft" for length, count in sorted(board_counts_2x2.items())])
    print(f"• 2x2 boards: {board_summary_2x2}")
    
    print("\nDETAILS - CUT PLANS:")
    print("-" * 40)
    print("2x4 BOARDS:")
    if result_2x4['success']:
        for plan in result_2x4['cutting_plan']:
            print(f"  Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n5/4\" x 6\" BOARDS:")
    if result_5x6['success']:
        for plan in result_5x6['cutting_plan']:
            print(f"  Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n1x4 BOARDS:")
    if result_gate['success']:
        for plan in result_gate['cutting_plan']:
            print(f"  Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n4x4 POSTS:")
    if result_4x4['success']:
        for plan in result_4x4['cutting_plan']:
            print(f"  Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    print(f"\n2x2 SECTIONS:")
    if result_2x2['success']:
        for plan in result_2x2['cutting_plan']:
            print(f"  Board {plan['board']} ({plan['board_length']//12}ft): pieces = {plan['pieces']} (total: {plan['total']}), leftover = {plan['leftover']}")
    else:
        print(f"  No solution found - try adjusting parameters")
    
    # Section-by-Section Build Guide
    print("\n" + "="*60)
    print("SECTION-BY-SECTION BUILD GUIDE")
    print("="*60)
    
    # Collect all section mappings from successful results
    all_sections = {}
    
    # Add 2x4 sections
    if result_2x4.get('success') and result_2x4.get('section_boards'):
        for section, boards_with_lengths in result_2x4['section_boards'].items():
            if section not in all_sections:
                all_sections[section] = {}
            all_sections[section]['2x4'] = boards_with_lengths
    
    # Add 5/4" x 6" sections  
    if result_5x6.get('success') and result_5x6.get('section_boards'):
        for section, boards_with_lengths in result_5x6['section_boards'].items():
            if section not in all_sections:
                all_sections[section] = {}
            all_sections[section]['5/4"x6"'] = boards_with_lengths
    
    # Add gate panel sections
    if result_gate.get('success') and result_gate.get('section_boards'):
        for section, boards_with_lengths in result_gate['section_boards'].items():
            if section not in all_sections:
                all_sections[section] = {}
            all_sections[section]['1x4'] = boards_with_lengths
    
    # Add 4x4 post sections
    if result_4x4.get('success') and result_4x4.get('section_boards'):
        for section, boards_with_lengths in result_4x4['section_boards'].items():
            if section not in all_sections:
                all_sections[section] = {}
            all_sections[section]['4x4'] = boards_with_lengths
    
    # Print section-by-section guide
    section_order = ['section_1', 'section_2', 'section_3', 'section_4', 'section_5', 'section_6', 'section_7', 
                     'section_5_6', 'section_6_7', 'gate_1', 'gate_2', 'gate_1_panels', 'gate_2_panels',
                     '60_inch_posts', '96_inch_posts']
    
    for section_name in section_order:
        if section_name in all_sections:
            print(f"\n{section_name.upper().replace('_', ' ')}:")
            section_data = all_sections[section_name]
            for lumber_type, boards_with_cuts in section_data.items():
                # Format as #board_num[cut_lengths]
                boards_list = []
                for board_num, cut_lengths in boards_with_cuts:
                    cuts_str = ",".join([str(cut) for cut in cut_lengths])
                    boards_list.append(f"#{board_num}[{cuts_str}]")
                boards_display = ", ".join(boards_list)
                board_count = len(boards_with_cuts)
                print(f"  {lumber_type}: {board_count} board(s) - {boards_display}")


if __name__ == "__main__":
    main()
