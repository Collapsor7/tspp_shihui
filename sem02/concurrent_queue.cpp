#include <iostream> 
#include <queue> 
#include <mutex> 
#include <condition_variable> 
#include <thread> 
#include <vector> 
#include <random> 
#include <chrono> 


template<typename T> 
class MyConcurrentQueue { 

public: 
	MyConcurrentQueue(size_t capacity) : capacity(capacity), done(false) {} 

	void put(int id, const std::vector<T>& items) { 
		std::unique_lock<std::mutex> lock(mtx); 

		cond_full.wait(lock, [this, &items] { return queue.size() + items.size() <= capacity; }); 
		
		for (const auto & item:items){
			queue.push(item);
		}
		
		std::cout << "Producer " << id << " produced "<< items.size() << " items, total items: " << queue.size() << std::endl; 
		cond_empty.notify_all(); 
	} 

	bool get(int id , T & item) { 
		std::unique_lock<std::mutex> lock(mtx); 

		cond_empty.wait(lock, [this] { return !queue.empty() || done; }); 
		if (queue.empty() && done){
			return false;
		}
		item =queue.front();
		queue.pop();
		std::cout << "Consumer " << id << " consumed 1 item, remaining items: " << queue.size() << std::endl; 
 
		cond_full.notify_all(); 
	
		return true;
	} 
	
	void set_done(){
		std::lock_guard<std::mutex> lock(mtx);
		done = true;
		cond_empty.notify_all();
	}
	
private: 
	std::queue<T> queue;
	int item_count;
	size_t capacity;
	std::mutex mtx; 
	std::condition_variable cond_full;
	std::condition_variable cond_empty;
	bool done;
}; 


void producer(MyConcurrentQueue<int>& q, int id) { 
	std::random_device rd; 
	std::mt19937 gen(rd()); 
	std::uniform_int_distribution<> dist(1, 5); 
	int produce_times = dist(gen); 
	std::cout << "Producer " << id << " will produce " << produce_times << " TIMES.\n"; 
	for (int i = 0; i < produce_times; ++i) { 
		std::vector<int> items(15,1);
		q.put(id, items); 

		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	} 
	std::cout << "Producer " << id << " finished producing." << std::endl; 
} 


void consumer(MyConcurrentQueue<int>& q, int id) { 
	int item;
	while (q.get(id,item)) { 
		std::this_thread::sleep_for(std::chrono::milliseconds(300)); 
	} 
	std::cout << "Consumer" << id << "finished consuming." << std::endl;
} 


void run_test(int num_producers, int num_consumers, int queue_capacity) { 
	MyConcurrentQueue<int> queue(queue_capacity); 

	std::vector<std::thread> producers; 
	std::vector<std::thread> consumers; 


	auto start_time = std::chrono::high_resolution_clock::now(); 

	for (int i = 0; i < num_producers; ++i) { 
		producers.emplace_back(producer, std::ref(queue), i + 1); 
	} 

	for (int i = 0; i < num_consumers; ++i) { 
		consumers.emplace_back(consumer, std::ref(queue), i + 1); 
	} 

	for (auto & thread : producers) { 
		thread.join(); 
	} 
	
	queue.set_done();
	
	for (auto & thread : consumers){
		thread.join();
	}

	auto end_time = std::chrono::high_resolution_clock::now(); 
	std::chrono::duration<double> duration = end_time - start_time; 


	std::cout << "Elapsed time: " << duration.count() << " seconds\n"; 

} 

int main() { 
	const int queue_capacity = 20; 

	std::cout << "Running test with multiple producers and consumers...\n"; 
 
	std::cout << "Test 1: {1 producer, N consumers}\n"; 
	run_test(1, 3, queue_capacity); 

	std::cout << "\nTest 2: {N producers, 1 consumer}\n"; 
	run_test(3, 1, queue_capacity); 

	std::cout << "\nTest 3: {M producers, N consumers}\n"; 
	run_test(2, 5, queue_capacity); 

	std::cout << "\nTest 4: {1 producer, 1 consumer}\n"; 
	run_test(1, 1, queue_capacity); 

return 0; 
}
