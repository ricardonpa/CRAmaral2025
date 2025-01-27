import sys
import torch
from pymatgen.core import Composition
from typing import Union

allowedElements = ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W", "Re", "Ru"]

# Define the predictors list with all input features in the correct order
predictors = [
    # Elements
    'Mo', 'Mn', 'W', 'Ta', 'Hf', 'Zr', 'Be', 'Cu', 'B', 'Cr', 
    'Al', 'Fe', 'Sn', 'Nb', 'U', 'Ti', 'V', 'Re', 'Ir', 'Bi', 
    'Si', 'Co', 'Ni', 'N', 'C', 'O',
    # Structure types
    'Structure_other', 'Structure_FCC', 'Structure_HCP', 'Structure_BCC',
    # Processing types
    'Processing_other', 'Processing_A', 'Processing_HIP', 'Processing_Q'
]

def callModel(input_data):
    # Convert DataFrame to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.FloatTensor(input_data.values).to(device)

    # Load the entire model
    model = torch.load('v2.pth')

    # Make prediction
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Post-process predictions
    logta = predictions[:, 0].cpu().numpy() * 100  # Multiply by 100 to revert scaling
    d0 = predictions[:, 1].cpu().numpy() ** 3     # Cube to revert transformation
    d1 = predictions[:, 2].cpu().numpy() ** 3     # Cube to revert transformation
    d2 = predictions[:, 3].cpu().numpy() ** 3     # Cube to revert transformation
    
    # Create results dictionary
    results = {
        'logta': logta[0],
        'd0': d0[0],
        'd1': d1[0],
        'd2': d2[0]
    }
    
    return results

def predict(
        comp: Union[Composition, str],
        structure_type = "?",
        processing_type = "?",
        outputType: str = "array") -> Union[dict, list]:
    
    assert isinstance(comp, (str, Composition)), "comp must be a string or a pymatgen Composition object."

    if isinstance(comp, str):
        comp = Composition(comp)

    elements = {}
    for element in comp:
        elStr = str(element)
        if elStr not in allowedElements:
            raise NotImplementedError(
                f"Element {elStr} passed as an input has not been implemented within Amaral2025 model covering: {', '.join(allowedElements)}.")
        elements[elStr] = comp.get_atomic_fraction(element)

    # Define composition_dict from elements
    composition_dict = {element: 0 for element in allowedElements}
    
    # Initialize all predictors to 0
    data = {predictor: [0] for predictor in predictors}
    
    # Fill in composition values
    for element, value in composition_dict.items():
        if element in data:
            data[element][0] = value
    
    # Set structure type (one-hot encoding)
    structure_col = f'Structure_{structure_type}'
    if structure_col in data:
        data[structure_col][0] = 1
    
    # Set processing type (one-hot encoding)
    processing_col = f'Processing_{processing_type}'
    if processing_col in data:
        data[processing_col][0] = 1
    
    # Create DataFrame
    input_data = pd.DataFrame(data)
    
    # Make prediction
    results = callModel(input_data)
    
    return results

if __name__ == "__main__":
    assert len(sys.argv) == 2
    print(predict(Composition(sys.argv[1]), outputType="array"))