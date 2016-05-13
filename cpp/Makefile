# Some Macros
# ---------------
# Compiler Name
CC		=	g++
# Compile Flags
CXXFLAGS=	-Wall -std=c++0x -fopenmp
# Include 
INCLUDES=	-I.	
# Libraries
LIBS	=	-lcblas -latlas
# Object Files
OBJS	=	main.o 
# Name of Executable
TARGET	=	test_main
# ---------------
all:	$(TARGET)
.cpp.o:
	$(CC) $(CXXFLAGS) -fPIC $(LDFLAGS) $(INCLUDES) -c $<

test_main:main.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) $< -o $@ -L/usr/lib64/atlas $(LIBS)

clean:
	-rm -f *.o core.* $(TARGET)
