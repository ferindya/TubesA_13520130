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

if __name__ == "__main__":
    file_name = file_name(str(input("Masukin namfel:")))
    #file_name = '../models/relu.json'
    case, expect = open_json(file_name)
    print(case)