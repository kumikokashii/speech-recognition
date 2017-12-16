
class Config():
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        sorted_attr = sorted(self.__dict__)
        sorted_attr.remove('name')
        sorted_attr.insert(0, 'name')
        
        output = ''
        for key in sorted_attr:
            value = self.__dict__[key]
            if key == 'add_noise':  # Avoid printing a list of numpy arrays
                value = (value[0], 'list of length {:,}'.format(len(value[1])))
            
            if isinstance(value, int):
                output += '{}: {:,}'.format(key, value)
            else:
                output += '{}: {}'.format(key, value)
            output += '\n'
        return output

