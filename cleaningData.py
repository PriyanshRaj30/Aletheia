import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer  # Swapped for efficiency
import warnings
warnings.filterwarnings('ignore')

def load_data():
    users_df = pd.read_csv('./Data/Users.csv', encoding='latin-1', on_bad_lines='skip')
    print("User columns:", users_df.columns.tolist())
    books_df = pd.read_csv('./Data/Books.csv', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    ratings_df = pd.read_csv('./Data/Ratings.csv', encoding='latin-1')
    print(f"Loaded Users: {users_df.shape}, Books: {books_df.shape}, Ratings: {ratings_df.shape}")
    return users_df, books_df, ratings_df

def clean_users(users_df):
    users_df['Location'] = users_df['Location'].fillna('unknown')
    users_df[['City', 'Country']] = users_df['Location'].str.split(', ', expand=True, n=1)
    users_df.drop('Location', axis=1, inplace=True)
    
    # Fix: Str for merge
    users_df['User-ID'] = users_df['User-ID'].astype(str)
    
    # Median impute + cap
    users_df['Age'] = pd.to_numeric(users_df['Age'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    users_df['Age'] = imputer.fit_transform(users_df[['Age']]).ravel()
    users_df['Age'] = np.clip(users_df['Age'], 5, 100)  # Single clip
    
    # Fix: Drop invalid countries (handle NaN)
    users_df = users_df[users_df['Country'].notna() & ~users_df['Country'].isin(['', 'unknown'])]
    
    print(f"Cleaned Users: {users_df.shape}")
    print(f"Age stats: Min={users_df['Age'].min()}, Max={users_df['Age'].max()}, Missing=0")
    return users_df

def clean_books(books_df):
    books_df['Book-Title'] = books_df['Book-Title'].str.strip()
    books_df['Book-Author'] = books_df['Book-Author'].str.strip()
    
    books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce')
    year_median = books_df['Year-Of-Publication'].median()
    books_df['Year-Of-Publication'].fillna(year_median, inplace=True)
    
    current_year = 2025
    books_df['Year-Of-Publication'] = np.clip(books_df['Year-Of-Publication'], 1900, current_year)
    
    # Fix: Early str + relaxed len (9-14 for safety)
    books_df['ISBN'] = books_df['ISBN'].astype(str).str.strip()
    valid_isbn = books_df['ISBN'].str.len().between(9, 14)
    pre_drop = len(books_df)
    books_df = books_df[valid_isbn]
    print(f"ISBN drop: {pre_drop - len(books_df)} rows")
    
    books_df.drop_duplicates(subset=['ISBN'], inplace=True)
    books_df = books_df[(books_df['Book-Title'] != '') & (books_df['Book-Author'] != '')]
    
    print(f"Cleaned Books: {books_df.shape}")
    print(f"Year stats: Min={books_df['Year-Of-Publication'].min()}, Max={books_df['Year-Of-Publication'].max()}, Missing=0")
    return books_df

def clean_ratings(ratings_df, min_rating_threshold=1):
    if min_rating_threshold > 0:
        ratings_df = ratings_df[ratings_df['Book-Rating'] >= min_rating_threshold]
    ratings_df.drop_duplicates(inplace=True)
    ratings_df['User-ID'] = ratings_df['User-ID'].astype(str)
    ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)
    print(f"Cleaned Ratings: {ratings_df.shape}")
    print(f"Rating distribution: {ratings_df['Book-Rating'].value_counts().sort_index()}")
    return ratings_df

# def merge_data(users_df, books_df, ratings_df):
#     combined_df = pd.merge(ratings_df, books_df, on='ISBN', how='left')
#     # Log unmatched
#     unmatched_books = 1 - len(combined_df) / len(ratings_df)
#     print(f"Unmatched books: {unmatched_books:.1%}")
    
#     combined_df = pd.merge(combined_df, users_df, on='User-ID', how='left')
#     combined_df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
    
#     print(f"Merged Data: {combined_df.shape}")
#     print(f"Missing values post-merge:\n{combined_df.isnull().sum()}")
    
#     user_item_matrix = combined_df.pivot_table(
#         index='User-ID', columns='ISBN', values='Book-Rating', fill_value=0
#     ).astype(np.float32)
    
#     sparsity = 1 - (user_item_matrix > 0).sum().sum() / user_item_matrix.size if user_item_matrix.size > 0 else np.nan
#     print(f"User-Item Matrix: {user_item_matrix.shape}")
#     print(f"Sparsity: {sparsity}")
    
#     return combined_df, user_item_matrix

from scipy.sparse import csr_matrix

def merge_data(users_df, books_df, ratings_df):
    # Merge ratings with books
    combined_df = pd.merge(ratings_df, books_df, on='ISBN', how='left')
    unmatched_books = 1 - len(combined_df) / len(ratings_df)
    print(f"Unmatched books: {unmatched_books:.1%}")
    
    # Merge with users
    combined_df = pd.merge(combined_df, users_df, on='User-ID', how='left')
    combined_df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
    print(f"Merged Data: {combined_df.shape}")
    print(f"Missing values post-merge:\n{combined_df.isnull().sum()}")
    
    # Encode User-ID and ISBN as categorical codes
    combined_df['user_code'] = combined_df['User-ID'].astype('category').cat.codes
    combined_df['book_code'] = combined_df['ISBN'].astype('category').cat.codes
    
    n_users = combined_df['user_code'].nunique()
    n_books = combined_df['book_code'].nunique()
    
    # Create sparse user-item matrix
    user_item_sparse = csr_matrix(
        (combined_df['Book-Rating'].values.astype(np.float32),
         (combined_df['user_code'].values, combined_df['book_code'].values)),
        shape=(n_users, n_books)
    )
    
    sparsity = 1 - (user_item_sparse.nnz / user_item_sparse.shape[0] / user_item_sparse.shape[1])
    print(f"Sparse User-Item Matrix: {user_item_sparse.shape}")
    print(f"Sparsity: {sparsity:.4%}")  # Use .nnz for count_nonzero    
    # Save mapping for later use
    user_mapping = dict(enumerate(combined_df['User-ID'].astype('category').cat.categories))
    book_mapping = dict(enumerate(combined_df['ISBN'].astype('category').cat.categories))
    
    return combined_df, user_item_sparse, user_mapping, book_mapping

def prepare_data():
    users_df, books_df, ratings_df = load_data()
    users_clean = clean_users(users_df.copy())
    books_clean = clean_books(books_df.copy())
    ratings_clean = clean_ratings(ratings_df.copy())
    # combined_df, user_item_matrix = merge_data(users_clean, books_clean, ratings_clean)
    # return combined_df, user_item_matrix
    combined_df, user_item_sparse, user_mapping, book_mapping = merge_data(users_clean, books_clean, ratings_clean)
    return combined_df, user_item_sparse, user_mapping, book_mapping


if __name__ == "__main__":
    # combined, matrix = prepare_data()
    # combined.to_csv('cleaned_books_data.csv', index=False)
    # matrix.to_pickle('user_item_matrix.pkl')
    # print("Data preparation complete!")
    # print(f"Final dataset shape: {combined.shape}")
    # print(f"Active users: {matrix.index.nunique()}, Books: {matrix.columns.nunique()}")

    combined, matrix, user_map, book_map = prepare_data()
    
    # Save combined CSV
    combined.to_csv('cleaned_books_data.csv', index=False)
    
    # Save sparse matrix
    from scipy import sparse
    sparse.save_npz('user_item_matrix_sparse.npz', matrix)
    
    # Save mappings
    import pickle
    with open('user_mapping.pkl', 'wb') as f:
        pickle.dump(user_map, f)
    with open('book_mapping.pkl', 'wb') as f:
        pickle.dump(book_map, f)
    
    print("Data preparation complete!")
    print(f"Final dataset shape: {combined.shape}")
    print(f"Active users: {matrix.shape[0]}, Books: {matrix.shape[1]}")