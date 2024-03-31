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
        if os.path.isfile('../models/'+path):
            return '../models/'+path
        else:
            print("Incorrect file name")
            return False

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