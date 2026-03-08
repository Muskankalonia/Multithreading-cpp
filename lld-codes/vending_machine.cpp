/*
The vending machine should support multiple products with different prices and quantities.
The machine should accept coins and notes of different denominations.
The machine should dispense the selected product and return change if necessary.
The machine should keep track of the available products and their quantities.
The machine should handle multiple transactions concurrently and ensure data consistency.
The machine should provide an interface for restocking products and collecting money.
The machine should handle exceptional scenarios, such as insufficient funds or out-of-stock products.
*/

#include <bits/stdc++.h>
using namespace std;

enum class denomination : int {
    dollar = 100,
    quater = 25,
    dime = 10,
    nickel = 5,
    penny = 1
};

struct product {
    string id;
    int price;
    int quantity;

    product(string id, int price, int quantity) : id(id), price(price), quantity(quantity) {}
};

class vendingMachine {
    mutex mtx;
    unordered_map<string, unique_ptr<product>> product_map;
    unordered_map<denomination, int> available_cash;

    public :
        static vendingMachine* instance() {
            static vendingMachine s;
            return &s;
        }

        void restock(unordered_map<string, pair<int, int>> &prd, unordered_map<denomination, int> cash) {
            lock_guard<mutex> lock(mtx);
            for(auto &[id, cur_prd] : prd) {
                if(product_map.find(id) != product_map.end()) {
                    product_map[id]->price = cur_prd.first;
                    product_map[id]->quantity = cur_prd.second;
                } else {
                    product_map[id] = make_unique<product> (id, cur_prd.first, cur_prd.second);
                }
            }

            for(auto &[i, quan] : cash) {
                available_cash[i] += quan;
            }
        }

        int purchase(string id, unordered_map<denomination, int> &cash) {
            lock_guard<mutex> lock(mtx);
            int total_cash = 0;
            for(auto &[den, quan] : cash) {
                total_cash += static_cast<int> (den) * quan;
            }

            product *prd = product_map[id].get();
            if(prd->quantity < 0 || total_cash < prd->price) return total_cash;
            
            pair<int, bool> change = calculateChange(cash, prd->price, total_cash);
            cout << change.second << endl;
            if(change.second) return change.first;
            return total_cash;
        }

        pair<int, bool> calculateChange(unordered_map<denomination, int> &cash, int price, int total_cash) {
            unordered_map <denomination, int> temp = available_cash;
            for(auto &[i, j] : cash) {
                temp[i] += j;
            }

            vector<denomination> denoms = {
                denomination::dollar,
                denomination::quater,
                denomination::dime,
                denomination::nickel,
                denomination::penny
            };

            int temp1 = total_cash - price;
            int i = 0;
            while((temp1 > 0) && (i < denoms.size())) {
                int pr = static_cast<int>(denoms[i]);
                while(temp1 >= pr) {
                    temp1 -= pr;
                    temp[denoms[i]]--;
                }
                i++;
            }

            if(temp1 == 0) {
                available_cash = temp;
                return {total_cash - price, true};
            }

            return {total_cash, false};
        }
};

int main() {
    vendingMachine * v = vendingMachine::instance();
    unordered_map<string, pair<int, int>> products;

    products["abc"] = {2, 5};
    products["bcd"] = {4, 3};

    unordered_map<denomination, int> cash;

    cash[denomination::penny] = 10;
    cash[denomination::dime] = 2;

    v->restock(products, cash);
    int change = v->purchase("abc", cash);
    cout << change;
}