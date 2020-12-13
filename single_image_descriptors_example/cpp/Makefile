SOURCES	= surf.cpp
OUT	= surf
CC	 = g++
LFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_xfeatures2d  -lopencv_features2d
CFLAGS	= `pkg-config --cflags opencv4`


all:
	$(CC) -o $(OUT) $(LFLAGS) $(CFLAGS) $(SOURCES)

