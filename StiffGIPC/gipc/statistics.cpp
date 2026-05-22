#include <gipc/statistics.h>
#include <fstream>
namespace gipc
{
Statistics::Statistics() {}

void Statistics::reset()
{
    m_json  = Json::object();
    m_frame = 0;
}

void Statistics::write_to_file(const std::string& filename)
{
    std::ofstream file(filename);
    file << m_json.dump(4);
}
}  // namespace gipc
