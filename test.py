string_list = ['abc', '   ', 'def', '', '  ghi  ', 'jkl', '']

# 使用列表推导式和 str.strip() 方法去除空字符串
new_string_list = [s.strip() for s in string_list if s.strip()]

# 输出结果
print(new_string_list)  # 输出：['abc', 'def', 'ghi', 'jkl']