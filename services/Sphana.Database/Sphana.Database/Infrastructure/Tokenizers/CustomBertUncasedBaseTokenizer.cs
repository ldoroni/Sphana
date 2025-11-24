using BERTTokenizers.Base;

namespace Sphana.Database.Infrastructure.Tokenizers
{
    public class CustomBertUncasedBaseTokenizer : UncasedTokenizer
    {
        public CustomBertUncasedBaseTokenizer(string vocabularyPath)
            : base(vocabularyPath)
        {
        }
    }
}