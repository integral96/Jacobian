cmake_minimum_required(VERSION 3.5)

project(untitled_ LANGUAGES CXX ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Boost ${BOOST_MIN_VERSION} COMPONENTS context system serialization filesystem date_time timer REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB SOURCES
    *.hpp
    *.cpp
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -pthread -lstdc++fs")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen ${Boost_LIBRARIES})
