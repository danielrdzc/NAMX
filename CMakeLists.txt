cmake_minimum_required(VERSION 3.21)

# ------------------------------------------------------------------------------
# 0. Auto-detectar vcpkg  (TIENE que correr ANTES de project())
#    Con esto ya no hace falta pasar -DCMAKE_TOOLCHAIN_FILE=... a mano.
#    Si no encuentra vcpkg (Linux / Raspberry Pi), sigue de largo y usa
#    pkg-config mas abajo.
# ------------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_vcpkg_candidates "")
    if(DEFINED ENV{VCPKG_ROOT})
        list(APPEND _vcpkg_candidates "$ENV{VCPKG_ROOT}")
    endif()
    list(APPEND _vcpkg_candidates
        "C:/dev/vcpkg"
        "$ENV{USERPROFILE}/vcpkg"
        "C:/vcpkg"
        "$ENV{HOME}/vcpkg"
        "/opt/vcpkg")

    foreach(_root IN LISTS _vcpkg_candidates)
        if(EXISTS "${_root}/scripts/buildsystems/vcpkg.cmake")
            set(CMAKE_TOOLCHAIN_FILE "${_root}/scripts/buildsystems/vcpkg.cmake"
                CACHE FILEPATH "vcpkg toolchain (auto-detectado)")
            message(STATUS "vcpkg encontrado en: ${_root}")
            break()
        endif()
    endforeach()

    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        message(STATUS "vcpkg no encontrado - se usara pkg-config para RtAudio")
    endif()
endif()

project(nam_pedal_proto CXX)

# NeuralAmpModelerCore requiere C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# En generadores de una sola config (Ninja / Makefiles, o sea el Pi):
# sin esto compila SIN optimizacion y te vas a comer xruns.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE no definido -> Release")
endif()

option(NAM_PEDAL_ENABLE_AVX2 "Compilar con AVX2 (x86-64 moderno). Apagalo si crashea con 'illegal instruction'." ON)
option(NAM_PEDAL_FAST_MATH  "Compilar con /fp:fast (MSVC) o -ffast-math. Apagalo para descartar artefactos de precision." ON)
option(NAM_PEDAL_ENABLE_A2_FAST "Usar el camino rapido especializado para modelos A2. Apagalo para caer al WaveNet generico." ON)

# ------------------------------------------------------------------------------
# 1. Fuentes de NAM
#    CONFIGURE_DEPENDS: si agregas/actualizas archivos del core, CMake se
#    re-configura solo en el siguiente build.
# ------------------------------------------------------------------------------
file(GLOB_RECURSE NAM_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore/NAM/*.cpp"
)

if(NAM_SOURCES STREQUAL "")
    message(FATAL_ERROR
        "No se encontraron fuentes en NeuralAmpModelerCore/NAM/. "
        "Falta clonar el submodulo?")
endif()

# ------------------------------------------------------------------------------
# 2. Ejecutable
# ------------------------------------------------------------------------------
add_executable(passthrough
    src/passthrough.cpp
    ${NAM_SOURCES}
)

# Al abrir el .slnx en Visual Studio, que arranque este proyecto y no ALL_BUILD
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT passthrough)

# ------------------------------------------------------------------------------
# 3. Includes y defines de NAM
# ------------------------------------------------------------------------------
target_include_directories(passthrough PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore/Dependencies/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore/Dependencies/nlohmann
)

target_compile_definitions(passthrough PRIVATE NAM_SAMPLE_FLOAT)

if(NAM_PEDAL_ENABLE_A2_FAST)
    target_compile_definitions(passthrough PRIVATE NAM_ENABLE_A2_FAST)
endif()

# ------------------------------------------------------------------------------
# 4. RtAudio
# ------------------------------------------------------------------------------
find_package(RtAudio CONFIG QUIET)
if(RtAudio_FOUND)
    # Ruta vcpkg (Windows) / paquetes CMake config en Linux-Mac
    target_link_libraries(passthrough PRIVATE RtAudio::rtaudio)

    # vcpkg marca sus includes como "system", y con MSVC eso sale como
    # /external:I en vez de /I. Si el compilador no resuelve
    # #include "RtAudio.h" por esa via, tira C1083. Los agregamos tambien
    # como include dirs normales para que no dependa de eso.
    get_target_property(_rtaudio_includes RtAudio::rtaudio INTERFACE_INCLUDE_DIRECTORIES)
    if(_rtaudio_includes)
        target_include_directories(passthrough PRIVATE ${_rtaudio_includes})
        message(STATUS "RtAudio includes: ${_rtaudio_includes}")
    endif()
else()
    # Fallback pkg-config (Linux/Mac via apt/brew)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(RTAUDIO REQUIRED rtaudio)
    target_include_directories(passthrough PRIVATE ${RTAUDIO_INCLUDE_DIRS})
    target_link_libraries(passthrough PRIVATE ${RTAUDIO_LIBRARIES})
    target_compile_options(passthrough PRIVATE ${RTAUDIO_CFLAGS_OTHER})
endif()

# ------------------------------------------------------------------------------
# 4b. libgpiod: botones fisicos. Opcional -- si no esta, se compila sin ellos.
# ------------------------------------------------------------------------------
if(NOT WIN32)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GPIOD QUIET libgpiod)
    endif()
    if(GPIOD_FOUND AND GPIOD_VERSION VERSION_GREATER_EQUAL 2.0)
        target_include_directories(passthrough PRIVATE ${GPIOD_INCLUDE_DIRS})
        target_link_libraries(passthrough PRIVATE ${GPIOD_LIBRARIES})
        target_compile_definitions(passthrough PRIVATE NAMX_HAS_GPIOD)
        message(STATUS "libgpiod ${GPIOD_VERSION} encontrado: botones habilitados")
    elseif(GPIOD_FOUND)
        message(WARNING "libgpiod ${GPIOD_VERSION} es muy vieja (hace falta 2.0+); "
                        "se compila sin botones")
    else()
        message(STATUS "libgpiod no encontrado: se compila sin botones "
                       "(sudo apt install libgpiod-dev)")
    endif()
endif()

# ------------------------------------------------------------------------------
# 5. Optimizacion y flags
# ------------------------------------------------------------------------------
if(MSVC)
    target_compile_options(passthrough PRIVATE
        /EHsc
        /MP                                          # compila en paralelo (mucho mas rapido)
        $<$<NOT:$<CONFIG:Debug>>:/O2>
    )
    if(NAM_PEDAL_FAST_MATH)
        target_compile_options(passthrough PRIVATE $<$<NOT:$<CONFIG:Debug>>:/fp:fast>)
    endif()
    if(NAM_PEDAL_ENABLE_AVX2)
        target_compile_options(passthrough PRIVATE $<$<NOT:$<CONFIG:Debug>>:/arch:AVX2>)
    endif()
    target_compile_definitions(passthrough PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(passthrough PRIVATE -O3)
    if(NAM_PEDAL_FAST_MATH)
        target_compile_options(passthrough PRIVATE -ffast-math)
    endif()
    if(NAM_PEDAL_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(passthrough PRIVATE -mavx2 -mfma)
    endif()
    if(NOT APPLE)
        target_link_libraries(passthrough PRIVATE stdc++fs)
    endif()
endif()

# ------------------------------------------------------------------------------
# 6. Copiar el modelo al lado del .exe despues de compilar
#    Asi el exe corre igual desde build/Release/ que desde la raiz del proyecto.
# ------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_model.nam")
    add_custom_command(TARGET passthrough POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/test_model.nam"
                "$<TARGET_FILE_DIR:passthrough>/test_model.nam"
        COMMENT "Copiando test_model.nam junto al ejecutable")
endif()

# Si algun dia creas una carpeta models/ (varios .nam + IRs), se copia entera.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
    add_custom_command(TARGET passthrough POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/models"
                "$<TARGET_FILE_DIR:passthrough>/models"
        COMMENT "Copiando models/ junto al ejecutable")
endif()

message(STATUS "----------------------------------------------------------")
message(STATUS "  nam_pedal_proto")
message(STATUS "  toolchain : ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "  generador : ${CMAKE_GENERATOR}")
message(STATUS "  fuentes NAM: ${CMAKE_CURRENT_SOURCE_DIR}/NeuralAmpModelerCore/NAM")
message(STATUS "----------------------------------------------------------")
