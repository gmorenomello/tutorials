cmake_minimum_required (VERSION 2.6) # defines the minimum required version
cmake_policy(SET CMP0012 NEW) # new policy of cmake
PROJECT(section2) #PROJECT TITLE
# Requires OpenCv
FIND_PACKAGE(OpenCv 3.0.0 REQUIRED)
# Show a message with the opencv version detected
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
# Create a variable called SRC
SET(SRC main.cpp)
# Create our executable
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC})
# LINK OUR LIBRARY
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})



add_executable(${PROJECT_NAME} main.cpp)

#create our hello library
# add_library(Hello hello.cpp hello.h)

# Create our application that uses our new library
# add_executable(executable main.cpp)

# link our executable with the new library
# target_link_libraries( executable Hello)

