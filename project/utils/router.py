from modules.drug_module import analyze_drug
from modules.job_module import analyze_job

def route_request(module_type: str, input_text: str):
    """
    Routes the input to the appropriate module based on module_type.
    """
    module_type = module_type.strip().lower()
    
    if module_type == "drug":
        print("Routing to Drug Analysis Module...")
        return analyze_drug(input_text)
    elif module_type == "job":
        print("Routing to Job Analysis Module...")
        return analyze_job(input_text)
    else:
        raise ValueError(f"Unknown module type '{module_type}'. Choose 'drug' or 'job'.")
