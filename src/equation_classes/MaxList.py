from bisect import bisect_left


class MaxList():
    def __init__(self, args):
        self.max_elements_in_list = args.max_elements_in_list
        self.max_list_keys = []  # smallest element is left
        self.max_list_state = []
        self.args = args

    def add(self, state, key):
        if len(self.max_list_keys) >= self.max_elements_in_list:
            if self.max_list_keys[0] >= key:
                return
            else:
                self.max_list_keys.pop(0)
                self.max_list_state.pop(0)

        index = bisect_left(a=self.max_list_keys, x=key)
        self.max_list_keys.insert(index, key)
        self.max_list_state.insert(index, state)
