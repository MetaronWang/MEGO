## turn on testing
enable_testing()

# define test
add_test(
        NAME
        IMP_Python_Test
        COMMAND
        ${CMAKE_COMMAND} -E env IMP_MODULE_PATH=$<TARGET_FILE_DIR:ComIMP>
        ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test.py
)