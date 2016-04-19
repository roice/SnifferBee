CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# name of this project
set(PRJ_NAME ppmencoder)

# change this line according to arm gcc toolchain installation
message("cmake system name is \"${CMAKE_SYSTEM}\"")
if(APPLE)
    set(TOOLCHAIN_PREFIX /Users/roice/workspace/stm32-tools/gcc-arm-embedded-4_8-q3/gcc-arm-none-eabi-4_8-2014q3) # for my Mac
    message("Toolchain prefix set to \"${TOOLCHAIN_PREFIX}\"")
endif()

# set stm32 cmake stuff
set(STM32_CHIP STM32F103C8)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/../../3rdparty/stm32-cmake/cmake/gcc_stm32.cmake)
set(STM32Cube_DIR ${CMAKE_SOURCE_DIR}/../../3rdparty/STM32Cube_FW_F1_V1.3.0)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/stm32-cmake/cmake)

# release or debug
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

# create project
project(${PRJ_NAME})

# enable asembly language for stm32 startup script
ENABLE_LANGUAGE(ASM)

# find CMSIS and HAL inc and src
FIND_PACKAGE(CMSIS REQUIRED)
FIND_PACKAGE(STM32HAL COMPONENTS rcc dma cortex gpio tim uart REQUIRED)

# include header file dir
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMSIS_INCLUDE_DIRS})
include_directories(${STM32HAL_INCLUDE_DIR})

# source files of this project
SET(PROJECT_SOURCES
    src/main.c src/stm32f1xx_hal_msp.c src/stm32f1xx_it.c
)

# compile
ADD_EXECUTABLE(${CMAKE_PROJECT_NAME}.elf ${PROJECT_SOURCES} ${CMSIS_SOURCES} ${STM32HAL_SOURCES})

STM32_SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME}.elf)
#STM32_ADD_HEX_BIN_TARGETS(${CMAKE_PROJECT_NAME}.elf)
ADD_CUSTOM_COMMAND(TARGET ${CMAKE_PROJECT_NAME}.elf POST_BUILD COMMAND ${CMAKE_OBJCOPY} ARGS -Oihex ${CMAKE_PROJECT_NAME}.elf ${CMAKE_PROJECT_NAME}.hex)
ADD_CUSTOM_COMMAND(TARGET ${CMAKE_PROJECT_NAME}.elf POST_BUILD COMMAND ${CMAKE_OBJCOPY} ARGS -Obinary ${CMAKE_PROJECT_NAME}.elf ${CMAKE_PROJECT_NAME}.bin)

# prepare .gdbinit file for debugging using GDB
# the input argument (first) is relative to the current source directory and
# the output argument (second) is relative to the current binary (i.e., build)
# directory.
configure_file(script/gdbinit .gdbinit COPYONLY)
