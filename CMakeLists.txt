cmake_minimum_required(VERSION 3.19)
project(RAGn_Roll_Indexer)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP)


add_executable(RAGn_Roll_Indexer D:/Boody/GP/Indexer/RAGn-Roll-Indexer/vamana/main.cpp)
#add_executable(RAGn_Roll_Indexer D:/Boody/GP/Indexer/RAGn-Roll-Indexer/kd-tree/kd-tree.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(RAGn_Roll_Indexer PUBLIC OpenMP::OpenMP_CXX)
endif()



find_package(Boost 1.82 REQUIRED COMPONENTS system filesystem serialization REQUIRED)
message("boost lib: ${Boost_LIBRARIES}, inc:${Boost_INCLUDE_DIR}")
target_link_libraries(RAGn_Roll_Indexer PUBLIC Boost::serialization Boost::filesystem Boost::system)
