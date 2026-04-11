from modules.drug_module import analyze_drug

def route_request(module_type: str, input_text: str):
    """
    Routes the input to the appropriate module based on module_type.
    """
    module_type = module_type.strip().lower()
    
    if module_type == "drug":
        print("Routing to Drug Analysis Module...")
        return analyze_drug(input_text)
    else:
        raise ValueError(f"Unknown module type '{module_type}'. Only 'drug' analysis is supported.")
