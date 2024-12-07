import json


def find_lowest_point(json_path=r'F:\Actual Projects\capstone_server_final\server_new\server_new\json_data\client_data.json'):

    z_points = list()

    with open(json_path) as data_file:
        data = json.load(data_file)
        for i in data.values():
            shapes = i
            break

    for i in shapes:
        z_points.append(i['y'])
    z_points.sort()
    print(z_points[0])
    return z_points[0]


print(find_lowest_point())
