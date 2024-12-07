import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
file = open(r"C:\Users\suhas\OneDrive\Desktop\server_new\recognized.txt", "a")
text = pytesseract.image_to_string(r'D:\capstone\capstone_test\easy.jpg')
print(text)
file.write(text)
file.write("$\n")
print("text written into file gg for now")
file.close 
