'''
Created on Dec 8, 2013

@author: RuiWang
'''

class EventsUtil:
    def to_Integer(self, val_str):
        int_rep = 0
        for i in range(0, 4):
            temp = ord(val_str[i])
            int_rep = int_rep << 8
            int_rep = int_rep + temp        

        return int_rep
    
    def to_Float(self, strr):
        '''
         int s = ((bits >> 31) == 0) ? 1 : -1;
         int e = ((bits >> 23) & 0xff);
         int m = (e == 0) ?
                 (bits & 0x7fffff) << 1 :
                 (bits & 0x7fffff) | 0x800000;
        s*m*2^(e-150)
        '''
        int_rep = self.to_Integer(strr)
        bits = int_rep
        
        s = -1
        if (bits >> 31) == 0:
            s = 1
        
        e = (bits >> 23) & 0xff
        
        if e == 0:
            m = (bits & 0x7fffff) << 1
        else:
            m = (bits & 0x7fffff) | 0x800000
            
        num = s * m * (2 ** (e - 150)) 
        # print num
        x = str(num)
        return x
    
    def to_Long(self, val_str):
        int_rep = 0
        for i in range(8):
            temp = ord(val_str[i])
            int_rep = int_rep << 8
            int_rep = int_rep + temp        
            
        return int_rep
    
    def to_Double(self, strr):
        int_rep = self.to_Long(strr)
        bits = int_rep
        s = -1
        if (bits >> 63) == 0:
            s = 1
        
        e = (bits >> 52) & 0x7ff
        
        if e == 0:
            m = (bits & 0xfffffffffffff) << 1
        else:
            m = (bits & 0xfffffffffffff) | 0x10000000000000
            
        num = s * m * (2 ** (e - 1075)) 
        # print num
        x = str(num)
        return x
