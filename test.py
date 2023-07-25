import re
s = '/usr/sbin/sendmail - 0 errors, 4 warnings'
m = re.match(r'(\S+) - (\d+) errors, (\d+) warnings', s)
print(m.groups())
