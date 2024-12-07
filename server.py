# This is to be run in the blender scripting console ONLY!
# This is the code that is in the test_model.blend

import socket
import bpy
import json

# factor is apercentage (so if you want o scale it to half size, you type in 50)


def scale_everything_down(factor):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(factor/100, factor/100, factor/100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(
        True, True, True), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)

    bpy.ops.object.select_all(action='DESELECT')


def server_program():
    # get the hostname
    host = socket.gethostname()
    port = 5000

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    z_lowest = list()

    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        data = conn.recv(1024).decode()
        if not data:
            # if data is not received break
            break
        data = str(data)
        shape, x, y, z, width, height = data.split(' ')
        # test_str = "\n\nshape:  " + shape + "\nx : " + x + "\ny :" + y + "\nz : " + z

        # print("from connected user: \nshape : " + shape +
        #   "\tdim : " + dim + "\tx : " + x, "\ty : ", y, "\tz : ", z)

        x = int(float((x)))
        y = int(float((y)))
        z = int(float((z)))
        width = int(float(width))
        height = int(float(height))

        dim = (height + width)//2

        lowest_point = find_lowest_of_shape(shape, y, dim)

        data_to_json = {
            "shape": shape,
            "dim": dim,
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "height": height,
            "lowest_point": lowest_point
        }

        write_json(data_to_json, json_obj_name="render_details",
                   filename=r"F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\json_data\ocr.json")

        # creating this list to find the lowest point of the render so we can add in afloor function spawn_floor(z)
        z_lowest.append(z)

        if shape.lower() == "cube":
            build_cube(dim, x, y, z)
            data = "Cube built!"
        elif shape.lower() == "sphere":

            build_sphere(dim, x, y, z)
            data = "Sphere built!"
        elif shape.lower() == "triangular_prism":

            build_ramp(dim, x, y, z,  width, height)
            data = "Ramp built!"
        elif shape.lower() == "pentagonal_prism":
            build_pent_prism(dim, x, y, z)
            data = "Pentagonal Prism built!"
        else:
            data = "shape not found!"
        conn.send(data.encode())  # send data to the client

    conn.close()  # close the connection

    # recivng simlarly for OCR values on the port 5001
    # get the hostname
    ocr_host = socket.gethostname()
    ocr_port = 5001

    ocr_server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    # bind host address and port together
    ocr_server_socket.bind((ocr_host, ocr_port))

    # configure how many client the server can listen simultaneously
    ocr_server_socket.listen(2)
    ocr_conn, ocr_address = ocr_server_socket.accept()  # accept new connection
    print("Connection from: " + str(ocr_address))

    while True:
        ocr_data = ocr_conn.recv(1024).decode()
        if not ocr_data:
            # if data is not received break
            break
        ocr_data = str(ocr_data)
        p = 0
        q = 0
        z = 0
        x, y, w, z, details, p = ocr_data.split('$')
        print(x, y, w, z, details)
        x = int(float((x)))
        y = int(float((y)))
        z = int(float((z)))
        build_char(x, y, z, details)
#        x , y , w , h , details  = data.split(' ')
#
#        ocr_data_to_json = {
#        "x" :x ,
#        "y" : y ,
#        "width" :w ,
#        "height" : h ,
#        "details" : details
#        }
        ocr_data_to_json = {
            "ocr_data": ocr_data,
            "ocr_data_type": str(type(ocr_data))
        }
        write_json(ocr_data_to_json, json_obj_name="ocr_details")

    ocr_conn.close()


def build_cube(size, x, y, z):  # , scale):
    y -= (size/2)
    x += (size/2)
    z -= (size/2)
    bpy.ops.mesh.primitive_cube_add(size=size, location=(x, y, z))
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'ACTIVE'
    bpy.ops.object.modifier_add(type='COLLISION')
#    s = scale


def build_sphere(size, x, y, z):  # , scale):
    #    bpy.ops.mesh.primitive_ico_sphere_add(radius=size, location=(x, y, z))
    size = size/2  # size is the diameter, we need the radius
    y -= (size)
    x += (size)
    z -= (size)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, z))
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'ACTIVE'
    bpy.ops.object.modifier_add(type='COLLISION')
#    s = scale
#    bpy.ops.transform.resize(value=(s/100 , s/100 , s/100))


def build_ramp(dim, x, y, z, w, h):  # , scale):
    verts = [(x, y, z), (x, y-h, z), (x+w, y-h, z),
             (x, y, z-dim), (x, y-h, z-dim), (x+w, y-h, z-dim)]
    faces = [(0, 1, 2), (3, 4, 5), (0, 3, 4, 1), (0, 3, 5, 2), (1, 4, 5, 2)]

    mesh = bpy.data.meshes.new("r")
    object = bpy.data.objects.new("r", mesh)

    bpy.context.collection.objects.link(object)
    mesh.from_pydata(verts, [], faces)

    bpy.context.view_layer.objects.active = object
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.ops.object.modifier_add(type='COLLISION')

#    bpy.context.space_data.context = 'CONSTRAINT'
#    bpy.context.space_data.context = 'PHYSICS'
# bpy.ops.object.effector_add(type='FORCE', location=(lx, ly, lz) line for adding forces


def cleanup():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')


def spawn_floor(x, y):
    x = 125*x/100
    if y < 0:
        bpy.ops.mesh.primitive_plane_add(
            size=x,
            calc_uvs=True,
            enter_editmode=False,
            align='WORLD',
            location=(0, -1*y, 0),
            rotation=(1.5708, 0, 0),
            scale=(0, 0, 0))
    elif y >= 0:
        bpy.ops.mesh.primitive_plane_add(
            size=x,
            calc_uvs=True,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            rotation=(1.5708, 0, 0),
            scale=(0, 0, 0))
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.ops.object.modifier_add(type='COLLISION')
    # floor must be spawned in y direction
    # if lowest point is above 0 then keep floor on 0, else keep it at lowest point and prompt user if additional height is required
#    bpy.ops.transform.resize(value=(.10 , .10 , .10))


def write_json(new_data, json_obj_name,  filename=r'F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\json_data\ocr.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[json_obj_name].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


def find_lowest_point(json_path=r'F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\json_data\ocr.json'):

    z_points = list()

    with open(json_path) as data_file:
        data = json.load(data_file)
        for i in data.values():
            shapes = i
            break

    for i in shapes:
        z_points.append(i['y'])
    z_points.sort()
#    print(z_points[0])
    return z_points[0]
    # return z_points[0]
    # since we are plotting stuff from the x and y axis the gravity must be in negative y direction because that is where the gravity is acting on the 2d image


def find_max_x(json_path=r'F:\Actual Projects\capstone_server_final\capstone_final_postreview3\capstone\json_data\ocr.json'):

    x_points = list()

    with open(json_path) as data_file:
        data = json.load(data_file)
        for i in data.values():
            shapes = i
            break

    for i in shapes:
        x_points.append(i['x'])
    x_points.sort()
#    print(x_points[-1])
    return x_points[-1]


def find_lowest_of_shape(shape, y, dim):
    if shape == "cube" or shape == "sphere":
        lowest_point = y
    elif shape == "triangular_prism" or shape == "pentagonal_prism":
        lowest_point = y

    return lowest_point


def build_char(x, y, z, details):
    z = 150
    font_curve = bpy.data.curves.new(type="FONT", name="numberPlate")
    font_curve.body = details
    obj = bpy.data.objects.new(name="text", object_data=font_curve)
    mat_red = bpy.data.materials.new("Text")
    mat_red.diffuse_color = (1,0,0,1)
    mat_red.metallic = 2
    if len(font_curve.materials) == 0:
        font_curve.materials.append(mat_red)
    else:
        font_curve.materials[0] = mat_red 
    obj.location = (x, y, z)
    obj.scale = (50, 50, 50)
    bpy.context.scene.collection.objects.link(obj)


def build_pent_prism(size, x, y, z):
    size = size / 2
    y -= (size)
    x += (size)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=5, radius=size, location=(x, y, -50), depth=100)
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'ACTIVE'
    bpy.ops.object.modifier_add(type='COLLISION')


def add_bg(path, x_off, y_off,  size):
    #    bpy.ops.object.empty_add(type='IMAGE', radius=size, align='VIEW', location=(x_off, y_off, 0), rotation=(0, -0, 0), scale=(1, 1, 1))
    bpy.ops.object.load_background_image(filepath=path)
    bpy.context.object.empty_display_size = size
    bpy.context.object.location[0] = x_off/2
    bpy.context.object.location[1] = y_off/2


def get_bg_details(path):

    with open(path, 'r') as bg_details:
        bg_data = json.load(bg_details)
        img_path = bg_data["image_details"][0]["img_path"]
        x_off = bg_data["image_details"][0]["img_width"]
        y_off = bg_data["image_details"][0]["img_height"]
        size = x_off

    return img_path, x_off, y_off, size


if __name__ == '__main__':
    cleanup()
    server_program()
    s = 25
    # scale_everything_down(s)
    spawn_floor(find_max_x()*4, find_lowest_point())
    img_path, x_off, y_off, size = get_bg_details(
        r'F:\Actual Projects\capstone_esa_review\capstone\json_data\shape.json')
    add_bg(img_path, x_off, y_off, size)
