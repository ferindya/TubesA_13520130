import json
import os

def file_name(path):
    '''Fungsi input untuk mengetahui apakah path file sesuai
	   input :  
			path: path to JSON file
	   return:  
			valid_path
	'''
    if os.path.isfile(path):
        return path
    else:
        if os.path.isfile('test/'+path):
            return 'test/'+path
        else:
            current_dir = os.path.dirname(__file__)
            module_dir = os.path.join(current_dir, '..', 'test', path)
            return module_dir

def open_json(file):
    '''Fungsi input untuk mengelola masukan file JSON
	   input :  
			file: JSON file
	   return:  
			case, expect
	'''
    # Read
    json_file = open(file, "r")
    json_read = json_file.read()

    # Parse
    json_data = json.loads(json_read)
    case = json_data.get("case")
    expect = json_data.get("expect")

    return case, expect

def create_json(case, expect, file_name):
    # Create the case object
    case_data = {
        "model": {
            "input_size": case["model"]["input_size"],
            "layers": [{"number_of_neurons": layer["number_of_neurons"], "activation_function": layer["activation_function"]} for layer in case["model"]["layers"]]
        },
        "input": case["input"],
        "weights": case["weights"],
        "target": case["target"],
        "learning_parameters": case["learning_parameters"]
    }

    # Create the expect object
    expect_data = {
        "stopped_by": expect["stopped_by"]
    }
    if "final_weights" in expect:
        expect_data["final_weights"] = expect["final_weights"]

    # Create the JSON structure
    data = {"case": case_data, "expect": expect_data}

    # Write data to JSON file
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

def update_json(file_name, new_weights):
    if not isinstance(new_weights, list):
        new_weights = new_weights.tolist()

    # Read the existing JSON file
    with open(file_name, "r") as json_file:
        data = json.load(json_file)

    # Update the weights in the data
    data["case"]["weights"] = new_weights

    # Write the updated data back to the JSON file
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    file_name = file_name(str(input("Masukin nama file:")))
    #file_name = '../models/relu.json'
    case, expect = open_json(file_name)
    print(case)