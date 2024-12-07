# Execution of the blender script

**Note: The files in server.py and the code in the python script of test_model.blend are the same code.**

### Step 1

Run the python script in test_blend.py



### Step 2

Run \_client.py in an IDE or no the terminal, whichever you prefer.

          **Note: Blender will be in a state of "Not Responding" when you run \_client.py. This is normal.**  



### Step 3
In the terminal running \_client.py, you will get a prompt resembling this 

`input is in format --------> 'shape' 'dim' 'x' 'y' 'z'`



### Step 4

Enter the input in the same format (shapes present as of now are sphere and cube)

eg: `cube 2 0 0 0`



### Step 5

You will continue to be prompted to input shapes. To end this and display the shapes constructed in Blender, type in `bye` in the input prompt. 



### Step 6

Blender will now exit its non responsive state, and should display all of the shapes that were to be constructed. Additionaly, the shape data has also been added to data.json for further use. 
