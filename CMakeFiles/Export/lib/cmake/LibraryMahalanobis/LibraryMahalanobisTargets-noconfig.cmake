#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libmahala" for configuration ""
set_property(TARGET libmahala APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(libmahala PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmahala.so.1.0.1"
  IMPORTED_SONAME_NOCONFIG "libmahala.so.1.0.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS libmahala )
list(APPEND _IMPORT_CHECK_FILES_FOR_libmahala "${_IMPORT_PREFIX}/lib/libmahala.so.1.0.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
