/*
The parking lot should have multiple levels, each level with a certain number of parking spots.
The parking lot should support different types of vehicles, such as cars, motorcycles, and trucks.
Each parking spot should be able to accommodate a specific type of vehicle.
The system should assign a parking spot to a vehicle upon entry and release it when the vehicle exits.
The system should track the availability of parking spots and provide real-time information to customers.
The system should handle multiple entry and exit points and support concurrent access.
*/

#include <bits/stdc++.h>
using namespace std;

enum class vehicleType {
    Car, 
    Truck,
    Bike
};

struct parkingSpot {
    mutex mtx;
    string vehicle_id;
    vehicleType vehicle;
    bool is_available;

    parkingSpot(vehicleType v) : vehicle(v), is_available(true), vehicle_id("") {}
};

struct level {
    vector<unique_ptr<parkingSpot>> parking_arr;
    string level_id;

    level(string l) : level_id(l) {}
};

class parkingLot {

    vector<unique_ptr<level>> parking_lot;

    unordered_map<string, parkingSpot*>unParkMap;
    mutex index_mtx;

    unordered_map<vehicleType, atomic<int>> available_spots;
    
    public :

        parkingLot(int levels, unordered_map<vehicleType, int> &mp) {

            for(auto const& [type, count] : mp) {
                available_spots[type] = count * levels;
            }

            for(int i = 0; i < levels; i++) {

                parking_lot.emplace_back([this, i, &mp]() {
                    auto l = make_unique<level>(to_string(i));

                    for(auto itr = mp.begin(); itr != mp.end(); itr++) {
                        for(int j = 0; j < itr->second; j++) {
                            l->parking_arr.emplace_back(make_unique<parkingSpot>(itr->first));
                        }
                    }

                    return l;
                }());
                
            }
        }

        bool park(vehicleType v, string vehicle_id) {

            for(auto &l : parking_lot) {
                
                for(auto &s : l->parking_arr)  {
                    unique_lock<mutex> lock(s->mtx);
                    if(s->is_available && s->vehicle == v) {
                        s->is_available = false;
                        s->vehicle_id = vehicle_id;

                        {
                            lock_guard<mutex> lock(index_mtx);
                            unParkMap[vehicle_id] = s.get();
                        }
                        available_spots[v]--;
                        return true;
                    }

                }
            }

            return false;
        }

        void unPark(string vehicle_id) {

            parkingSpot *s = nullptr;
            
            {
                lock_guard<mutex> lock(index_mtx);
                if(unParkMap.find(vehicle_id) != unParkMap.end()) {
                    s = unParkMap[vehicle_id];
                    unParkMap.erase(vehicle_id);
                }

            }

            if(s) {
                unique_lock<mutex> lock(s->mtx);
                s->is_available = true;
                s->vehicle_id = "";
                available_spots[s->vehicle]++;
            }

        }

        int numAvaialbeSpots(vehicleType v) {
            return available_spots[v].load();
        }
};


int main() {
    
    unordered_map<vehicleType, int> mp;
    mp[vehicleType::Car] = 3;
    mp[vehicleType::Truck] = 4;
    mp[vehicleType::Bike] = 5;

    parkingLot parking_lot(3, mp);

    parking_lot.park(vehicleType::Car, "abc");
    parking_lot.park(vehicleType::Bike, "asdfs");

    cout << parking_lot.numAvaialbeSpots(vehicleType::Car) << endl;
    parking_lot.unPark("abc");

    cout << parking_lot.numAvaialbeSpots(vehicleType::Car) << endl;
}