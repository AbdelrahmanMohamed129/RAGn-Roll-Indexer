cmake_minimum_required(VERSION 3.19)
project(RAGn_Roll_Indexer)

set(CMAKE_CXX_STANDARD 17)
#SET(CMAKE_CXX_FLAGS "-std=c++20")
find_package(OpenMP)

#add_executable(RAGn_Roll_Indexer D:/Boody/GP/Indexer/RAGn-Roll-Indexer/kd-tree/kd-tree.cpp)

SET (BOOST_ROOT "C:/Program Files/boost_1_70_0")
SET (Boost_INCLUDE_DIRS "C:/Program Files/boost/include/boost-1_70")
SET (Boost_LIBRARIES "C:/Program Files/boost/lib/libboost_serialization-mgw13-mt-x64-1_70.a")
find_package(Boost 1.55.0 COMPONENTS serialization REQUIRED)

if (NOT Boost_FOUND)
message(FATAL_ERROR "Fatal error: Boost (version >= 1.55) required.")
else()
 message(STATUS "Setting up BOOST")
 message(STATUS " Includes - ${Boost_INCLUDE_DIRS}")
 message(STATUS " Library  - ${Boost_LIBRARIES}")

 include_directories(${Boost_INCLUDE_DIRS})
 add_executable(RAGn_Roll_Indexer D:/Boody/GP/Indexer/RAGn-Roll-Indexer/vamana/main.cpp)

 if(OpenMP_CXX_FOUND)
     target_link_libraries(RAGn_Roll_Indexer PUBLIC ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
 endif()

endif()