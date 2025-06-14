use ipc_lib::IPC;

#[repr(C)]
#[derive(Debug, Default)]
struct Data {
    counter: u32,
    value: f32,
}
fn main() {
    match IPC::<Data>::new("/my_topic") {
        Ok(shm) => {
            let data = shm.get();
            println!("Read: {:?}", data);
        }
        Err(e) => {
            eprintln!("Failed to read shared memory: {}", e);
        }
    }
}