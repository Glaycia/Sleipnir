macro(sleipnir_compiler_flags target)
  if (NOT MSVC)
    target_compile_options(${target} PRIVATE -Wall -pedantic -Wextra -Werror -Wno-unused-parameter)
  else()
    target_compile_options(${target} PRIVATE /wd4146 /wd4244 /wd4251 /wd4267 /WX)
  endif()
endmacro()
