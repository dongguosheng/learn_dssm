# Some Macros
# ---------------
# Compiler Name
CC		=	g++
# Compile Flags
CXXFLAGS=	-Wall -std=c++0x
# Include 
INCLUDES=	
# Libraries
LIBS	=	-lcblas -latlas
# Object Files
OBJS	=	Main.o 
# Name of Executable
TARGET	=	testMain
# ---------------
all:	$(TARGET)
.cpp.o:
	$(CC) $(CXXFLAGS) -fPIC $(LDFLAGS) $(INCLUDES) -c $<

testMain:Main.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) $< -o $@ -L/usr/lib64/atlas $(LIBS)

clean:
	-rm -f *.o core.* $(TARGET)
